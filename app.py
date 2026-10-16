"""
NEXT 2026 — Aplicação do Professor
Avaliação de modelos YOLO dos grupos, auditoria visual e geração de ranking.

Como usar:
  Instalar python 3.12, criar venv, atualizar pip, instalar requirements
  python app.py

Acesse: http://127.0.0.1:5000

Senha do professor (uma vez): python app.py --senha  e copie a linha para o .env
"""

import os
import hmac
import json
import pickletools
import shutil
import sys
import sqlite3
import tempfile
import threading
import time
import unicodedata
import urllib.request
import uuid
import webbrowser
import zipfile
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, Response, session
from werkzeug.security import check_password_hash, generate_password_hash
from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv

# =========================
# CONFIGURAÇÕES
# =========================

CONFIDENCE_MINIMA = 0.50  # confiança mínima para a detecção contar
CONFIDENCE_PREDICT = 0.25  # limiar do detector: mais baixo para o telão mostrar "quase lá"
DURACAO_AVALIACAO_PADRAO = 120  # segundos do turno, usado até o desafio ser cadastrado
INTERVALO_ENTRE_FRAMES = 0.05
TOP_PREDICOES_POR_CLASSE = 5  # evidências guardadas por classe
INTERVALO_EVIDENCIA_POR_CLASSE = 3.0  # evita salvar muitas imagens quase iguais
SEGUNDOS_CAMERA_INSTAVEL = 1.5   # sem frame por mais que isso, avisa na tela
MAX_EXTENSAO_TURNO_S = 30        # teto do tempo devolvido por queda de camera
LIMITE_CACHE_MODELOS = 2         # modelo do grupo atual + o do proximo
ABRIR_JANELAS_AO_INICIAR = True

CLASSES_COCO_PADRAO = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
}

# =========================
# CÓDIGO
# =========================

load_dotenv()  # lê o .env da pasta do app

app = Flask(__name__)
# Chave das sessões do Flask, definida no .env (veja .env.example).
# Sem ela, gera uma chave aleatória: funciona, mas as sessões caem a cada reinício.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

# Só o próprio notebook acessa o app: o telão abre aqui e é espelhado no
# projetor. As câmeras não precisam chegar ao app, é ele quem busca o /capture.
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = 5000

# Teto do upload de best.pt (um YOLO nano tem ~5 MB; sobra para modelos maiores).
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024

# Cookie de sessão fora do alcance de JavaScript e de outros sites.
# A sessão dura um dia de evento inteiro, para não deslogar no meio de um turno.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)

# Login único do professor. Sem ADMIN_PASSWORD_HASH no .env o app roda sem
# login (como antes) e avisa no console: o evento nunca fica travado.
ADMIN_USER = os.environ.get("ADMIN_USER", "professor")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")
LOGIN_ATIVO = bool(ADMIN_PASSWORD_HASH)

PASTA_MODELOS     = "modelos_recebidos"
PASTA_ANOTADAS    = "imagens_anotadas"
ARQUIVO_RESULTADOS = "dados/resultados.json"
ARQUIVO_BANCO = "dados/avaliacao.sqlite"
ARQUIVO_DESAFIO = "dados/desafio.json"
PASTA_BACKUPS = "dados/backups"

# Lista oficial de classes. Ela mora dentro de aluno/ porque essa pasta e
# entregue compactada aos grupos: o script de autoteste do aluno precisa
# achar o arquivo sem depender de nada fora da propria pasta.
# A tela /desafio edita este mesmo arquivo: nao existe copia paralela.
# Fora do monorepo (app clonado sozinho) a pasta aluno/ nao existe, entao
# o arquivo fica em dados/.
PASTA_ALUNO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "aluno"
))
if os.path.isdir(PASTA_ALUNO):
    ARQUIVO_CLASSES_OFICIAIS = os.path.join(PASTA_ALUNO, "classes-oficiais.txt")
else:
    ARQUIVO_CLASSES_OFICIAIS = os.path.join("dados", "classes-oficiais.txt")

# cria as pastas se não existirem
for pasta in [PASTA_MODELOS, PASTA_ANOTADAS, "dados"]:
    os.makedirs(pasta, exist_ok=True)

AVALIACOES_AO_VIVO = {}
AVALIACOES_LOCK = threading.Lock()
TELAO_JOB_ATIVO_ID = None

# Último frame anotado do turno, em memória. O telão recebe como MJPEG,
# igual ao /stream do ESP32: sem gravar em disco e sem esperar o polling.
FRAME_TELAO = {"jpeg": None, "contador": 0}
FRAME_TELAO_SINAL = threading.Condition()


def normalizar_nome(nome):
    """Remove acentos e espaços para usar como nome de pasta."""
    nome = unicodedata.normalize("NFD", nome)
    nome = "".join(c for c in nome if unicodedata.category(c) != "Mn")
    nome = nome.lower().replace(" ", "_").replace("-", "_")
    nome = "".join(c for c in nome if c.isalnum() or c == "_")
    return nome


def ler_json(caminho, padrao):
    """Lê um JSON local e retorna um padrão se o arquivo ainda não existir."""
    if not os.path.exists(caminho):
        return padrao
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return json.load(arquivo)
    except (json.JSONDecodeError, OSError):
        return padrao


def salvar_json(caminho, dados):
    """Salva um JSON local de forma legível."""
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)


def desafio_padrao():
    """Identidade do evento usada até o professor cadastrar o desafio."""
    return {
        "nome_evento": "NEXT 2026",
        "subtitulo": "Arena Drone - Tecnologia em IA",
        "duracao_s": DURACAO_AVALIACAO_PADRAO
    }


def ler_desafio():
    """Nome, subtítulo e duração do turno. As classes ficam em classes-oficiais.txt."""
    desafio = desafio_padrao()
    desafio.update(ler_json(ARQUIVO_DESAFIO, {}))
    return desafio


def salvar_desafio(nome_evento, subtitulo, duracao_s, classes):
    """Grava a identidade do evento e regrava a lista oficial de classes."""
    salvar_json(ARQUIVO_DESAFIO, {
        "nome_evento": nome_evento,
        "subtitulo": subtitulo,
        "duracao_s": duracao_s
    })
    with open(ARQUIVO_CLASSES_OFICIAIS, "w", encoding="utf-8", newline="\n") as arquivo:
        arquivo.write(texto_classes_oficiais(nome_evento, classes))


def texto_classes_oficiais(nome_evento, classes):
    """Monta o classes-oficiais.txt, o mesmo arquivo que vai para os alunos."""
    cabecalho = [
        f"# Classes oficiais do desafio {nome_evento}",
        "#",
        "# Regras:",
        "# - uma classe por linha",
        "# - letras minúsculas, sem acentos, sem espaços",
        "# - todos os grupos devem usar exatamente estes nomes",
        "# - a ORDEM é recomendada, não obrigatória: o app casa por nome.",
        "#   Ordem diferente gera aviso, nunca reprovação.",
        "#",
        "# Linhas que começam com # são comentários e são ignoradas.",
        "# Gerado pela tela /desafio do app do professor.",
        ""
    ]
    return "\n".join(cabecalho + classes) + "\n"


def normalizar_classes(texto):
    """Uma classe por linha: minúsculas, sem acento, sem espaço e sem repetição."""
    classes = []
    for linha in texto.splitlines():
        classe = normalizar_nome(linha.strip())
        if classe and classe not in classes:
            classes.append(classe)
    return classes


MODELOS_CARREGADOS = {}
MODELOS_LOCK = threading.Lock()


def detectar_dispositivo():
    """Descobre se há GPU utilizável. Sem GPU, cai para CPU sem reclamar."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", "CPU"


DISPOSITIVO, NOME_DISPOSITIVO = detectar_dispositivo()


# Um best.pt é um zip com um pickle dentro, e carregar um pickle pode executar
# código. Antes do YOLO() lemos o pickle SEM executá-lo e listamos o que ele
# importa: um modelo legítimo só traz torch, ultralytics e tipos básicos.
PICKLE_MODULOS_PERMITIDOS = {
    "torch", "ultralytics", "collections", "numpy", "_codecs",
    "builtins", "__builtin__", "pathlib", "datetime", "copyreg", "argparse",
}
# dentro de builtins, só tipos inofensivos (nada de eval, exec, getattr, open...)
PICKLE_BUILTINS_PERMITIDOS = {
    "set", "frozenset", "slice", "range", "complex", "bytearray", "bytes",
    "dict", "list", "tuple", "int", "float", "str", "bool", "object",
}
# partes de torch/ultralytics/numpy que baixam, instalam ou carregam código
PICKLE_PREFIXOS_PROIBIDOS = (
    "torch.hub", "torch.load", "torch.save", "torch.jit", "torch.package",
    "torch.ops", "torch._ops", "torch.utils.cpp_extension", "torch.serialization",
    "ultralytics.hub", "ultralytics.utils.downloads", "ultralytics.utils.checks",
    "numpy.load", "numpy.ctypeslib", "numpy.distutils", "numpy.f2py",
)


def globais_do_pickle(dados):
    """Lista os (módulo, nome) que um pickle importaria, sem executá-lo."""
    memo, textos, encontrados = {}, [], set()
    for opcode, arg, _ in pickletools.genops(dados):
        nome = opcode.name
        if nome in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                    "SHORT_BINSTRING", "BINSTRING", "STRING"):
            textos.append(arg)
        elif nome in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[arg] = textos[-1] if textos else None
        elif nome == "MEMOIZE":
            memo[len(memo)] = textos[-1] if textos else None
        elif nome in ("BINGET", "LONG_BINGET", "GET"):
            if isinstance(memo.get(arg), str):
                textos.append(memo[arg])
        elif nome in ("GLOBAL", "INST"):
            encontrados.add(tuple(arg.split(" ", 1)))
        elif nome == "STACK_GLOBAL":
            if len(textos) < 2:
                raise ValueError("pickle com importação ilegível")
            encontrados.add((textos[-2], textos[-1]))
    return encontrados


def verificar_pickle_modelo(caminho_modelo):
    """Recusa o best.pt se o pickle importar algo fora da lista permitida."""
    try:
        with zipfile.ZipFile(caminho_modelo) as arquivo_zip:
            pickles = [n for n in arquivo_zip.namelist() if n.endswith(".pkl")]
            if not pickles:
                raise ValueError("o arquivo não contém os dados do modelo")
            importados = set()
            for nome in pickles:
                importados |= globais_do_pickle(arquivo_zip.read(nome))
    except zipfile.BadZipFile:
        raise ValueError("o arquivo não é um best.pt do PyTorch/Ultralytics")
    except ValueError:
        raise
    except Exception as erro:
        raise ValueError(f"não foi possível inspecionar o arquivo ({erro})")

    suspeitos = []
    for modulo, nome in sorted(importados):
        completo = f"{modulo}.{nome}"
        raiz = modulo.split(".")[0]
        if raiz not in PICKLE_MODULOS_PERMITIDOS:
            suspeitos.append(completo)
        elif raiz in ("builtins", "__builtin__") and nome not in PICKLE_BUILTINS_PERMITIDOS:
            suspeitos.append(completo)
        elif any(completo == p or completo.startswith(p + ".") or modulo == p
                 for p in PICKLE_PREFIXOS_PROIBIDOS):
            suspeitos.append(completo)

    if suspeitos:
        print(f"[seguranca] best.pt recusado ({caminho_modelo}): {', '.join(suspeitos)}")
        raise ValueError("arquivo recusado por segurança: ele tenta usar "
                         + ", ".join(suspeitos[:5]))


def obter_modelo(caminho_modelo):
    """Carrega o best.pt usando cache quente, para a troca de grupo ser rápida.

    O cache guarda poucos modelos de propósito: cada YOLO carregado ocupa VRAM,
    e a GTX 1650 tem 4 GB. Guardar os vinte modelos do evento daria erro de
    memória no meio da competição.
    """
    with MODELOS_LOCK:
        if caminho_modelo in MODELOS_CARREGADOS:
            return MODELOS_CARREGADOS[caminho_modelo]

    verificar_pickle_modelo(caminho_modelo)
    modelo = YOLO(caminho_modelo)

    # inferência de aquecimento: paga o custo da primeira chamada agora,
    # e não no primeiro frame do grupo
    try:
        modelo.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                       conf=CONFIDENCE_PREDICT, device=DISPOSITIVO, verbose=False)
    except Exception:
        pass

    with MODELOS_LOCK:
        MODELOS_CARREGADOS[caminho_modelo] = modelo
        while len(MODELOS_CARREGADOS) > LIMITE_CACHE_MODELOS:
            MODELOS_CARREGADOS.pop(next(iter(MODELOS_CARREGADOS)))
    return modelo


def preaquecer_modelo(caminho_modelo):
    """Carrega um modelo em segundo plano, enquanto o turno anterior corre."""
    if not caminho_modelo or not os.path.exists(caminho_modelo):
        return
    threading.Thread(target=obter_modelo, args=(caminho_modelo,), daemon=True).start()


def ler_classes_oficiais():
    """Lê a lista oficial de classes do desafio, ignorando comentários e linhas vazias."""
    if not os.path.exists(ARQUIVO_CLASSES_OFICIAIS):
        return []

    classes = []
    with open(ARQUIVO_CLASSES_OFICIAIS, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha and not linha.startswith("#"):
                classes.append(linha.lower())
    return classes


def classes_oficiais_detalhadas():
    """Lista oficial com o ID do YOLO ao lado de cada classe."""
    return [
        {"id_classe_oficial": indice, "classe": classe}
        for indice, classe in enumerate(ler_classes_oficiais())
    ]


def comparar_com_classes_oficiais(classes_modelo):
    """Compara as classes de um best.pt com a lista oficial e devolve o veredito.

    A comparação é por nome, não por posição: o app casa as detecções usando
    modelo.names, então ordem diferente vira aviso e nunca reprovação.
    """
    oficiais = ler_classes_oficiais()
    encontradas = [str(classe).strip().lower() for classe in classes_modelo]

    linhas = []
    for posicao, esperada in enumerate(oficiais):
        presente = esperada in encontradas
        linhas.append({
            "id_oficial": posicao,
            "classe": esperada,
            "situacao": "ok" if presente else "faltando",
            "id_modelo": encontradas.index(esperada) if presente else ""
        })

    faltando = [linha["classe"] for linha in linhas if linha["situacao"] == "faltando"]
    extras = [classe for classe in encontradas if classe not in oficiais]
    ordem_diferente = not faltando and not extras and encontradas != oficiais

    if not oficiais:
        status = "sem_referencia"
        mensagem = "Nenhuma classe cadastrada. Cadastre as classes na tela Desafio."
    elif faltando or extras:
        status = "reprovado"
        partes = []
        if faltando:
            partes.append(f"faltam {len(faltando)}: {', '.join(faltando)}")
        if extras:
            partes.append(f"sobram {len(extras)}: {', '.join(extras)}")
        mensagem = "As classes não batem com a lista oficial — " + "; ".join(partes) + "."
    elif ordem_diferente:
        status = "aprovado_com_aviso"
        mensagem = (f"As {len(oficiais)} classes estão corretas, mas em ordem diferente da oficial. "
                    "O app casa por nome, então o modelo pode competir normalmente.")
    else:
        status = "aprovado"
        mensagem = f"As {len(oficiais)} classes batem exatamente com a lista oficial."

    return {
        "status": status,
        "mensagem": mensagem,
        "aprovado": status in {"aprovado", "aprovado_com_aviso"},
        "oficiais": oficiais,
        "encontradas": encontradas,
        "linhas": linhas,
        "faltando": faltando,
        "extras": extras,
        "ordem_diferente": ordem_diferente
    }


def listar_classes_do_modelo(caminho_modelo):
    """Carrega um best.pt e retorna as classes informadas pelo modelo."""
    verificar_pickle_modelo(caminho_modelo)
    modelo = YOLO(caminho_modelo)
    nomes = modelo.names

    if isinstance(nomes, dict):
        classes = [nomes[chave] for chave in sorted(nomes)]
    else:
        classes = list(nomes)

    return [str(classe) for classe in classes]


def resumir_modelo(caminho_modelo):
    """Retorna um resumo simples para o professor validar o best.pt recebido."""
    classes = listar_classes_do_modelo(caminho_modelo)
    classes_normalizadas = {classe.strip().lower() for classe in classes}
    parece_coco = len(classes_normalizadas) >= 60 and CLASSES_COCO_PADRAO.issubset(classes_normalizadas)

    validacao = comparar_com_classes_oficiais(classes)

    avisos = []
    if not classes:
        avisos.append("O modelo carregou, mas não informou classes.")
    if parece_coco:
        avisos.append("As classes parecem ser do COCO padrão. Confira se o grupo não enviou um modelo pronto.")
    if validacao["status"] == "sem_referencia":
        avisos.append("Não foi possível conferir as classes oficiais: cadastre as classes na tela Desafio.")

    return {
        "total_classes": len(classes),
        "classes": classes,
        "parece_coco": parece_coco,
        "avisos": avisos,
        "validacao": validacao
    }


def conectar_banco():
    """Abre o banco SQLite local."""
    conexao = sqlite3.connect(ARQUIVO_BANCO)
    conexao.row_factory = sqlite3.Row
    return conexao


def iniciar_banco():
    """Cria as tabelas do app se ainda não existirem."""
    with conectar_banco() as conexao:
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS grupos (
                grupo TEXT PRIMARY KEY,
                nome_exibir TEXT,
                turma TEXT,
                integrantes TEXT,
                modelo TEXT,
                data_hora TEXT,
                atualizado_em TEXT
            )
        """)
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS avaliacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grupo TEXT,
                rodada INTEGER,
                data_hora TEXT,
                imagem TEXT,
                classe_esperada TEXT,
                classe_detectada TEXT,
                confidence REAL,
                acertou INTEGER,
                tempo_ms REAL,
                modelo TEXT,
                confidence_minima REAL,
                imagem_anotada TEXT,
                stream_url TEXT,
                duracao_s INTEGER,
                frames_processados INTEGER,
                deteccoes INTEGER,
                confidence_minima_detectada REAL,
                confidence_maxima_detectada REAL
            )
        """)
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                tipo TEXT,
                grupo TEXT,
                detalhe TEXT
            )
        """)
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS evidencias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grupo TEXT,
                rodada INTEGER,
                data_hora TEXT,
                classe_detectada TEXT,
                confidence REAL,
                tempo_ms REAL,
                imagem_anotada TEXT,
                frame_indice INTEGER,
                status_revisao TEXT DEFAULT 'pendente'
            )
        """)
        colunas = [
            linha["name"]
            for linha in conexao.execute("PRAGMA table_info(avaliacoes)").fetchall()
        ]
        novas_colunas = {
            "stream_url": "TEXT",
            "duracao_s": "INTEGER",
            "frames_processados": "INTEGER",
            "deteccoes": "INTEGER",
            "confidence_minima_detectada": "REAL",
            "confidence_maxima_detectada": "REAL"
        }
        for coluna, tipo in novas_colunas.items():
            if coluna not in colunas:
                conexao.execute(f"ALTER TABLE avaliacoes ADD COLUMN {coluna} {tipo}")

        colunas_evidencias = [
            linha["name"]
            for linha in conexao.execute("PRAGMA table_info(evidencias)").fetchall()
        ]
        if "status_revisao" not in colunas_evidencias:
            conexao.execute("ALTER TABLE evidencias ADD COLUMN status_revisao TEXT DEFAULT 'pendente'")


def fazer_backup_do_banco():
    """Copia o banco inteiro antes de qualquer limpeza."""
    os.makedirs(PASTA_BACKUPS, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = os.path.join(PASTA_BACKUPS, f"avaliacao_{carimbo}.sqlite")
    shutil.copy2(ARQUIVO_BANCO, destino)
    return destino


def limpar_competicao(escopo):
    """Apaga os dados de ensaio antes do evento. Sempre faz backup antes.

    escopo "resultados": limpa evidências e avaliações, mas mantém
    os grupos e os modelos já cadastrados.
    escopo "tudo": limpa também os grupos.

    As imagens em imagens_anotadas/ não são apagadas: sem linha no banco elas
    deixam de aparecer nas telas, e manter o arquivo custa pouco e evita perda.
    """
    backup = fazer_backup_do_banco()

    with conectar_banco() as conexao:
        contagens = {}
        for tabela in ("evidencias", "avaliacoes"):
            contagens[tabela] = conexao.execute(f"SELECT COUNT(*) FROM {tabela}").fetchone()[0]
            conexao.execute(f"DELETE FROM {tabela}")
        if escopo == "tudo":
            contagens["grupos"] = conexao.execute("SELECT COUNT(*) FROM grupos").fetchone()[0]
            conexao.execute("DELETE FROM grupos")

    resumo = ", ".join(f"{quantidade} {tabela}" for tabela, quantidade in contagens.items())
    registrar_evento("reset_competicao", "", f"Escopo {escopo}: apagados {resumo}. Backup em {backup}")
    return backup, contagens


def registrar_evento(tipo, grupo, detalhe):
    """Registra ações importantes para auditoria."""
    with conectar_banco() as conexao:
        conexao.execute("""
            INSERT INTO eventos (data_hora, tipo, grupo, detalhe)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), tipo, grupo, detalhe))


def salvar_grupo(grupo):
    """Cadastra ou atualiza um grupo."""
    agora = datetime.now().isoformat()
    with conectar_banco() as conexao:
        existente = conexao.execute(
            "SELECT grupo FROM grupos WHERE grupo = ?",
            (grupo["grupo"],)
        ).fetchone()

        if existente:
            conexao.execute("""
                UPDATE grupos
                SET nome_exibir = ?, turma = ?, integrantes = ?, modelo = ?, atualizado_em = ?
                WHERE grupo = ?
            """, (
                grupo.get("nome_exibir", grupo["grupo"]),
                grupo.get("turma", ""),
                grupo.get("integrantes", ""),
                grupo.get("modelo", ""),
                agora,
                grupo["grupo"]
            ))
        else:
            conexao.execute("""
                INSERT INTO grupos
                (grupo, nome_exibir, turma, integrantes, modelo, data_hora, atualizado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                grupo["grupo"],
                grupo.get("nome_exibir", grupo["grupo"]),
                grupo.get("turma", ""),
                grupo.get("integrantes", ""),
                grupo.get("modelo", ""),
                grupo.get("data_hora", agora),
                agora
            ))


def proxima_rodada(grupo):
    """Retorna o número da próxima rodada de avaliação do grupo."""
    with conectar_banco() as conexao:
        linha = conexao.execute(
            "SELECT MAX(rodada) AS ultima FROM avaliacoes WHERE grupo = ?",
            (grupo,)
        ).fetchone()
    return (linha["ultima"] or 0) + 1


def salvar_avaliacoes(grupo, modelo, avaliacoes, evidencias=None, stream_url="", duracao_s=0, frames_processados=0, rodada=None):
    """Salva uma rodada de avaliação sem apagar rodadas anteriores."""
    if rodada is None:
        rodada = proxima_rodada(grupo)
    evidencias = evidencias or []

    with conectar_banco() as conexao:
        for avaliacao in avaliacoes:
            conexao.execute("""
                INSERT INTO avaliacoes
                (grupo, rodada, data_hora, imagem, classe_esperada, classe_detectada,
                 confidence, acertou, tempo_ms, modelo, confidence_minima, imagem_anotada,
                 confidence_minima_detectada, confidence_maxima_detectada)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                grupo,
                rodada,
                avaliacao["data_hora"],
                avaliacao["imagem"],
                avaliacao["classe_esperada"],
                avaliacao["classe_detectada"],
                avaliacao["confidence"],
                avaliacao["acertou"],
                avaliacao["tempo_ms"],
                modelo,
                CONFIDENCE_MINIMA,
                avaliacao.get("imagem_anotada", ""),
                avaliacao.get("confidence_minima_detectada", avaliacao["confidence"]),
                avaliacao.get("confidence_maxima_detectada", avaliacao["confidence"])
            ))
            conexao.execute("""
                UPDATE avaliacoes
                SET stream_url = ?, duracao_s = ?, frames_processados = ?, deteccoes = ?
                WHERE id = last_insert_rowid()
            """, (
                stream_url,
                duracao_s,
                frames_processados,
                avaliacao.get("deteccoes", 0)
            ))

        for evidencia in evidencias:
            conexao.execute("""
                INSERT INTO evidencias
                (grupo, rodada, data_hora, classe_detectada, confidence,
                 tempo_ms, imagem_anotada, frame_indice, status_revisao)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                grupo,
                rodada,
                evidencia["data_hora"],
                evidencia["classe_detectada"],
                evidencia["confidence"],
                evidencia["tempo_ms"],
                evidencia["imagem_anotada"],
                evidencia["frame_indice"],
                "pendente"
            ))
    registrar_evento("avaliacao_camera", grupo, f"Rodada {rodada}: {len(avaliacoes)} objetos, {frames_processados} frames")


def avaliacoes_ultima_rodada(grupo):
    """Retorna as avaliações da última rodada do grupo."""
    with conectar_banco() as conexao:
        linha = conexao.execute(
            "SELECT MAX(rodada) AS ultima FROM avaliacoes WHERE grupo = ?",
            (grupo,)
        ).fetchone()
        ultima = linha["ultima"]
        if ultima is None:
            return []

        linhas = conexao.execute("""
            SELECT rodada, data_hora, imagem, classe_esperada, classe_detectada,
                   confidence, acertou, tempo_ms, stream_url, duracao_s,
                   frames_processados, deteccoes,
                   confidence_minima_detectada, confidence_maxima_detectada
            FROM avaliacoes
            WHERE grupo = ? AND rodada = ?
            ORDER BY id
        """, (grupo, ultima)).fetchall()

    return [dict(linha) for linha in linhas]


def listar_historico():
    """Lista todas as avaliações salvas, da mais recente para a mais antiga."""
    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT grupo, rodada, data_hora, imagem, classe_esperada,
                   classe_detectada, confidence, acertou, tempo_ms, modelo,
                   stream_url, duracao_s, frames_processados, deteccoes,
                   confidence_minima_detectada, confidence_maxima_detectada
            FROM avaliacoes
            ORDER BY id DESC
        """).fetchall()
    return [dict(linha) for linha in linhas]


def listar_eventos():
    """Lista eventos importantes do app."""
    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT data_hora, tipo, grupo, detalhe
            FROM eventos
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()
    return [dict(linha) for linha in linhas]


def listar_evidencias_agrupadas():
    """Lista as evidências visuais agrupadas por grupo, rodada e classe."""
    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT e.id, e.grupo, g.nome_exibir, e.rodada, e.data_hora,
                   e.classe_detectada, e.confidence, e.tempo_ms,
                   e.imagem_anotada, e.frame_indice, e.status_revisao
            FROM evidencias e
            LEFT JOIN grupos g ON g.grupo = e.grupo
            ORDER BY e.rodada DESC, e.data_hora DESC, e.grupo, e.classe_detectada
        """).fetchall()

    grupos = {}
    for linha in linhas:
        item = dict(linha)
        chave = (
            item["grupo"],
            item["nome_exibir"] or item["grupo"],
            item["rodada"]
        )
        if chave not in grupos:
            grupos[chave] = {
                "grupo": item["grupo"],
                "nome_exibir": item["nome_exibir"] or item["grupo"],
                "rodada": item["rodada"],
                "data_hora": item["data_hora"],
                "classes": {}
            }

        classe = item["classe_detectada"]
        if classe not in grupos[chave]["classes"]:
            grupos[chave]["classes"][classe] = {
                "classe_detectada": classe,
                "itens": []
            }
        grupos[chave]["classes"][classe]["itens"].append(item)

    resultado = []
    for grupo in grupos.values():
        classes = []
        for classe in grupo["classes"].values():
            classe["itens"].sort(key=lambda item: item["confidence"], reverse=True)
            classe["pendentes"] = sum(1 for item in classe["itens"] if item["status_revisao"] == "pendente")
            classes.append(classe)
        classes.sort(key=lambda item: item["classe_detectada"])
        grupo["classes"] = classes
        # rodada auditada = nenhuma imagem pendente (o selo na tela usa isto)
        grupo["total_itens"] = sum(len(classe["itens"]) for classe in classes)
        grupo["pendentes"] = sum(classe["pendentes"] for classe in classes)
        resultado.append(grupo)

    resultado.sort(key=lambda item: (item["rodada"], item["data_hora"]), reverse=True)
    return resultado


def listar_evidencias_do_grupo(grupo, limite=6):
    """Retorna as melhores evidências de um grupo para o dashboard."""
    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT id, classe_detectada, confidence, imagem_anotada, status_revisao
            FROM evidencias
            WHERE grupo = ?
            ORDER BY rodada DESC, confidence DESC
            LIMIT ?
        """, (grupo, limite)).fetchall()
    return [dict(linha) for linha in linhas]


def listar_melhores_evidencias_por_classe(grupo):
    """Retorna a melhor evidência de cada classe na última rodada do grupo."""
    with conectar_banco() as conexao:
        linha = conexao.execute(
            "SELECT MAX(rodada) AS ultima FROM evidencias WHERE grupo = ?",
            (grupo,)
        ).fetchone()
        ultima = linha["ultima"]
        if ultima is None:
            return []

        linhas = conexao.execute("""
            SELECT id, classe_detectada, confidence, imagem_anotada, status_revisao
            FROM evidencias
            WHERE grupo = ? AND rodada = ?
              AND status_revisao = 'validada'
            ORDER BY classe_detectada, confidence DESC
        """, (grupo, ultima)).fetchall()

    melhores = {}
    for linha in linhas:
        item = dict(linha)
        classe = item["classe_detectada"]
        melhores.setdefault(classe, item)

    return sorted(melhores.values(), key=lambda item: (-item["confidence"], item["classe_detectada"]))


def atualizar_status_evidencia(evidencia_id, status):
    """Marca uma evidência como validada, rejeitada ou pendente."""
    if status not in {"validada", "rejeitada", "pendente"}:
        return

    with conectar_banco() as conexao:
        linha = conexao.execute(
            "SELECT grupo, classe_detectada FROM evidencias WHERE id = ?",
            (evidencia_id,)
        ).fetchone()
        if not linha:
            return
        conexao.execute(
            "UPDATE evidencias SET status_revisao = ? WHERE id = ?",
            (status, evidencia_id)
        )

    registrar_evento("revisao_evidencia", linha["grupo"], f"{linha['classe_detectada']} -> {status}")


def normalizar_camera_url(url):
    """Aceita IP/base e garante que a URL final termine com /capture."""
    url = url.strip().rstrip("/")
    if url.endswith("/capture"):
        return url
    if url.endswith("/stream"):
        return url[:-len("/stream")] + "/capture"
    return url + "/capture"


def buscar_frame_http(camera_url):
    """Busca uma foto JPEG no ESP32-CAM e decodifica para OpenCV."""
    with urllib.request.urlopen(camera_url, timeout=5) as resposta:
        bytes_imagem = resposta.read()

    array = np.frombuffer(bytes_imagem, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Imagem inválida recebida do ESP32-CAM.")
    return frame


def iniciar_captura(camera_url, parar):
    """Busca frames sem parar e guarda apenas o mais recente.

    Desacoplar captura de inferência muda o gargalo: em vez de somar os dois
    tempos por ciclo, o FPS passa a ser o do mais lento dos dois. Frame antigo
    é descartado de propósito — no telão, latência baixa vale mais que
    processar todos os frames.
    """
    estado = {"frame": None, "contador": 0, "ultimo_ok": time.time(), "erros": 0}
    trava = threading.Lock()

    def laco():
        while not parar.is_set():
            inicio_ciclo = time.time()
            try:
                frame = buscar_frame_http(camera_url)
            except Exception:
                with trava:
                    estado["erros"] += 1
                time.sleep(0.2)
                continue

            with trava:
                estado["frame"] = frame
                estado["contador"] += 1
                estado["ultimo_ok"] = time.time()

            # INTERVALO_ENTRE_FRAMES e piso do ciclo, nao espera adicional:
            # com camera real a captura ja passa disso e nao dorme nada.
            resto = INTERVALO_ENTRE_FRAMES - (time.time() - inicio_ciclo)
            if resto > 0:
                time.sleep(resto)

    thread = threading.Thread(target=laco, daemon=True)
    thread.start()
    return estado, trava, thread


def guardar_candidato_evidencia(candidatos, chave, evidencia):
    """Guarda candidatos separados por classe para escolher as melhores no fim."""
    if chave not in candidatos:
        candidatos[chave] = []
    candidatos[chave].append(evidencia)


def salvar_top_evidencias(candidatos, pasta_anotadas):
    """Salva no disco as melhores evidências de cada classe."""
    evidencias = []

    for chave in sorted(candidatos):
        itens = candidatos[chave]
        melhores = sorted(itens, key=lambda item: item["confidence"], reverse=True)[:TOP_PREDICOES_POR_CLASSE]

        for posicao, item in enumerate(melhores, start=1):
            nome_classe = normalizar_nome(item["classe_detectada"])
            conf_int = int(item["confidence"] * 1000)
            nome_arquivo = f"{nome_classe}_{posicao:02d}_{conf_int:04d}.jpg"
            caminho_absoluto = os.path.join(pasta_anotadas, nome_arquivo)
            cv2.imwrite(caminho_absoluto, item["imagem"])

            caminho_relativo = os.path.relpath(caminho_absoluto, PASTA_ANOTADAS).replace("\\", "/")
            evidencias.append({
                "data_hora": item["data_hora"],
                "classe_detectada": item["classe_detectada"],
                "confidence": round(item["confidence"], 4),
                "tempo_ms": round(item["tempo_ms"], 1),
                "imagem_anotada": caminho_relativo,
                "frame_indice": item["frame_indice"]
            })

    return evidencias


def avaliar_camera(modelo, camera_url, duracao_s, pasta_anotadas, classes_esperadas,
                   callback_ao_vivo=None, ao_status_camera=None):
    """Roda o turno de um grupo pedindo imagens JPEG ao ESP32-CAM pelo /capture.

    Do START ao fim do tempo, todo frame vai para o telão e todas as classes
    esperadas do desafio contam. Classe fora da lista aparece desenhada na
    imagem, mas não pontua nem gera evidência.

    Queda de câmera não consome o turno do grupo: o tempo parado é devolvido
    ao final, até o teto de MAX_EXTENSAO_TURNO_S.
    """
    camera_url = normalizar_camera_url(camera_url)
    classes_esperadas = set(classes_esperadas)

    os.makedirs(pasta_anotadas, exist_ok=True)

    inicio = time.time()
    fim_previsto = inicio + duracao_s
    frames_processados = 0
    tempos = []
    estatisticas = {}
    candidatos = {}
    ultimo_candidato_por_classe = {}
    ultimo_contador = 0
    camera_caida = False
    tempo_devolvido = 0.0
    inicio_da_falha = 0.0

    parar_captura = threading.Event()
    captura, trava_captura, thread_captura = iniciar_captura(camera_url, parar_captura)

    try:
        while time.time() < fim_previsto:
            with trava_captura:
                frame = captura["frame"]
                contador = captura["contador"]
                ultimo_ok = captura["ultimo_ok"]
                total_erros_camera = captura["erros"]

            parado_s = time.time() - ultimo_ok

            if parado_s >= SEGUNDOS_CAMERA_INSTAVEL:
                if not camera_caida:
                    camera_caida = True
                    inicio_da_falha = ultimo_ok
                if ao_status_camera:
                    ao_status_camera(False, parado_s, total_erros_camera)
            elif camera_caida:
                # o tempo sem câmera não pode consumir o turno do grupo
                devolver = min(time.time() - inicio_da_falha,
                               max(0.0, MAX_EXTENSAO_TURNO_S - tempo_devolvido))
                fim_previsto += devolver
                tempo_devolvido += devolver
                camera_caida = False
                if ao_status_camera:
                    ao_status_camera(True, 0.0, total_erros_camera)

            # sem frame novo ainda: espera de leve em vez de reprocessar o mesmo
            if frame is None or contador == ultimo_contador:
                time.sleep(0.005)
                continue

            ultimo_contador = contador
            agora = time.time()

            # o detector roda com limiar baixo; o 0.50 é filtro da regra, mais adiante
            inicio_inferencia = time.time()
            resultados = modelo.predict(frame, conf=CONFIDENCE_PREDICT,
                                        device=DISPOSITIVO, verbose=False)
            tempo_ms = (time.time() - inicio_inferencia) * 1000
            tempos.append(tempo_ms)
            frames_processados += 1
            classes_frame = []
            img_anotada = frame.copy()

            for resultado in resultados:
                img_anotada = resultado.plot()

                for box in resultado.boxes:
                    classe_id = int(box.cls[0])
                    nome_classe = str(modelo.names[classe_id]).strip().lower()
                    conf = float(box.conf[0])

                    if nome_classe not in classes_esperadas:
                        continue

                    classes_frame.append({
                        "classe": nome_classe,
                        "confidence": round(conf, 4)
                    })

                    # abaixo do mínimo aparece no telão como "quase lá", mas não conta
                    if conf < CONFIDENCE_MINIMA:
                        continue

                    if nome_classe not in estatisticas:
                        estatisticas[nome_classe] = {
                            "deteccoes": 0,
                            "soma_conf": 0.0,
                            "melhor_conf": 0.0,
                            "menor_conf": 1.0
                        }

                    estatisticas[nome_classe]["deteccoes"] += 1
                    estatisticas[nome_classe]["soma_conf"] += conf
                    if conf > estatisticas[nome_classe]["melhor_conf"]:
                        estatisticas[nome_classe]["melhor_conf"] = conf
                    if conf < estatisticas[nome_classe]["menor_conf"]:
                        estatisticas[nome_classe]["menor_conf"] = conf

                    ultimo_candidato = ultimo_candidato_por_classe.get(nome_classe, 0)
                    if agora - ultimo_candidato >= INTERVALO_EVIDENCIA_POR_CLASSE:
                        guardar_candidato_evidencia(candidatos, nome_classe, {
                            "data_hora": datetime.now().isoformat(),
                            "classe_detectada": nome_classe,
                            "confidence": conf,
                            "tempo_ms": tempo_ms,
                            "frame_indice": frames_processados,
                            "imagem": img_anotada.copy()
                        })
                        ultimo_candidato_por_classe[nome_classe] = agora

            if callback_ao_vivo:
                callback_ao_vivo(
                    img_anotada,
                    frames_processados,
                    tempo_ms,
                    estatisticas,
                    classes_frame,
                    duracao_s - max(0.0, fim_previsto - time.time()),
                    duracao_s
                )

    finally:
        parar_captura.set()
        thread_captura.join(timeout=2)

    tempo_medio = sum(tempos) / len(tempos) if tempos else 0
    avaliacoes = []
    evidencias = salvar_top_evidencias(candidatos, pasta_anotadas)
    melhor_evidencia_por_classe = {}
    for evidencia in evidencias:
        classe = evidencia["classe_detectada"]
        if classe not in melhor_evidencia_por_classe:
            melhor_evidencia_por_classe[classe] = evidencia["imagem_anotada"]

    if not estatisticas:
        avaliacoes.append({
            "data_hora": datetime.now().isoformat(),
            "imagem": "capture",
            "classe_esperada": "",
            "classe_detectada": "nenhuma",
            "confidence": 0.0,
            "acertou": 0,
            "tempo_ms": round(tempo_medio, 1),
            "imagem_anotada": "",
            "deteccoes": 0,
            "confidence_minima_detectada": 0.0,
            "confidence_maxima_detectada": 0.0
        })
        return avaliacoes, evidencias, frames_processados

    for classe, dados_classe in sorted(estatisticas.items()):
        confidence_media = dados_classe["soma_conf"] / dados_classe["deteccoes"]
        avaliacoes.append({
            "data_hora": datetime.now().isoformat(),
            "imagem": "capture",
            "classe_esperada": "",
            "classe_detectada": classe,
            "confidence": round(confidence_media, 4),
            "acertou": 1,
            "tempo_ms": round(tempo_medio, 1),
            "imagem_anotada": melhor_evidencia_por_classe.get(classe, ""),
            "deteccoes": dados_classe["deteccoes"],
            "confidence_minima_detectada": round(dados_classe["menor_conf"], 4),
            "confidence_maxima_detectada": round(dados_classe["melhor_conf"], 4)
        })

    return avaliacoes, evidencias, frames_processados


def ler_resultados():
    """Lê os grupos e a última avaliação salva no SQLite."""
    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT grupo, nome_exibir, turma, integrantes, modelo, data_hora
            FROM grupos
            ORDER BY data_hora
        """).fetchall()

    grupos = []
    for linha in linhas:
        grupo = dict(linha)
        grupo["avaliacoes"] = avaliacoes_ultima_rodada(grupo["grupo"])
        grupos.append(grupo)

    return {"grupos": grupos}


def salvar_resultados(dados):
    """Mantido apenas para compatibilidade com versões antigas."""
    for grupo in dados.get("grupos", []):
        salvar_grupo(grupo)


def migrar_json_para_sqlite():
    """Importa resultados.json antigo, se existir e o banco ainda estiver vazio."""
    if not os.path.exists(ARQUIVO_RESULTADOS):
        return

    with conectar_banco() as conexao:
        total = conexao.execute("SELECT COUNT(*) AS total FROM grupos").fetchone()["total"]
    if total > 0:
        return

    with open(ARQUIVO_RESULTADOS, "r", encoding="utf-8") as f:
        dados = json.load(f)

    for grupo in dados.get("grupos", []):
        salvar_grupo(grupo)
        avaliacoes = grupo.get("avaliacoes", [])
        if avaliacoes:
            salvar_avaliacoes(grupo["grupo"], grupo.get("modelo", ""), avaliacoes)

    if dados.get("grupos"):
        registrar_evento("migracao", "", "Dados importados de resultados.json")


def buscar_grupo(dados, nome_grupo):
    """Retorna o grupo pelo nome ou None se não encontrar."""
    for grupo in dados["grupos"]:
        if grupo["grupo"] == nome_grupo:
            return grupo
    return None


def resultado_do_grupo(grupo, avaliacoes, classes_esperadas):
    """Resultado de um grupo na última rodada dele.

    Cada classe esperada vale a maior confidence VALIDADA na auditoria.
    A pontuação é a média só das classes que o grupo conseguiu mostrar:
    classe que não saiu no dado não pune ninguém.
    """
    rodada = avaliacoes[0]["rodada"] if avaliacoes else None
    deteccoes_por_classe = {
        item["classe_detectada"]: item.get("deteccoes") or 0
        for item in avaliacoes
        if item["classe_detectada"] in classes_esperadas
    }

    with conectar_banco() as conexao:
        linhas = conexao.execute("""
            SELECT classe_detectada, confidence, status_revisao
            FROM evidencias
            WHERE grupo = ? AND rodada = ?
        """, (grupo, rodada)).fetchall()

    melhor_validada = {}
    contagem = {"validada": 0, "rejeitada": 0, "pendente": 0}
    for linha in linhas:
        if linha["classe_detectada"] not in classes_esperadas:
            continue
        status = linha["status_revisao"] or "pendente"
        contagem[status] = contagem.get(status, 0) + 1
        if status == "validada":
            classe = linha["classe_detectada"]
            melhor_validada[classe] = max(melhor_validada.get(classe, 0.0), linha["confidence"])

    classes = [
        {
            "classe": classe,
            "confidence": round(melhor_validada.get(classe, 0.0), 4),
            "pontuou": classe in melhor_validada,
            "deteccoes": deteccoes_por_classe.get(classe, 0)
        }
        for classe in classes_esperadas
    ]
    valores = list(melhor_validada.values())

    return {
        "rodada": rodada,
        "classes": classes,
        "classes_pontuadas": len(valores),
        "pontuacao": round(sum(valores) / len(valores), 4) if valores else 0.0,
        "tem_pontuacao": bool(valores),
        "deteccoes_total": sum(deteccoes_por_classe.values()),
        "evidencias_validadas": contagem["validada"],
        "evidencias_rejeitadas": contagem["rejeitada"],
        "evidencias_pendentes": contagem["pendente"]
    }


def calcular_ranking(dados):
    """Classificação dos grupos avaliados.

    Ordem: maior pontuação (média das classes validadas), mais classes
    pontuadas, mais detecções no turno e menor tempo médio de inferência.
    Grupo sem nenhuma evidência validada fica no fim, sem pontuação.
    """
    classes_esperadas = ler_classes_oficiais()
    ranking = []

    for grupo in dados["grupos"]:
        avaliacoes = grupo.get("avaliacoes", [])
        if not avaliacoes:
            continue

        tempo_medio = sum(a.get("tempo_ms", 0) for a in avaliacoes) / len(avaliacoes)
        ranking.append({
            "grupo": grupo["grupo"],
            "nome_exibir": grupo.get("nome_exibir") or grupo["grupo"],
            "turma": grupo.get("turma", ""),
            "frames_processados": avaliacoes[0].get("frames_processados") or 0,
            "tempo_medio_ms": round(tempo_medio, 1),
            **resultado_do_grupo(grupo["grupo"], avaliacoes, classes_esperadas)
        })

    ranking.sort(key=lambda item: (
        0 if item["tem_pontuacao"] else 1,
        -item["pontuacao"],
        -item["classes_pontuadas"],
        -item["deteccoes_total"],
        item["tempo_medio_ms"]
    ))

    for posicao, item in enumerate(ranking, start=1):
        item["posicao"] = posicao

    return ranking


def resumo_classes_ao_vivo(estatisticas, classes_esperadas):
    """Uma linha por classe esperada, na ordem do desafio, para o telão e o painel."""
    classes = []
    for classe in classes_esperadas:
        dados_classe = estatisticas.get(classe)
        classes.append({
            "classe": classe,
            "deteccoes": dados_classe["deteccoes"] if dados_classe else 0,
            "melhor_conf": round(dados_classe["melhor_conf"], 4) if dados_classe else 0.0
        })
    return classes


def atualizar_job_ao_vivo(job_id, **dados):
    """Atualiza o estado de uma avaliação em andamento."""
    with AVALIACOES_LOCK:
        if job_id in AVALIACOES_AO_VIVO:
            AVALIACOES_AO_VIVO[job_id].update(dados)


def atualizar_status_camera(job_id, camera_ok, parado_s, total_erros):
    """Publica a saúde da câmera. O último frame válido continua na tela."""
    if camera_ok:
        mensagem = ""
    else:
        mensagem = (f"CÂMERA INSTÁVEL — {parado_s:.0f}s sem imagem. "
                    f"O tempo parado será devolvido ao turno.")

    with AVALIACOES_LOCK:
        job = AVALIACOES_AO_VIVO.get(job_id)
        if not job:
            return
        ja_avisado = not job.get("camera_ok", True)
        job["camera_ok"] = camera_ok
        job["camera_mensagem"] = mensagem
        job["camera_erros"] = total_erros
        grupo = job["grupo"]

    if not camera_ok and not ja_avisado:
        registrar_evento("camera_instavel", grupo,
                         f"Sem imagem por {parado_s:.1f}s ({total_erros} falhas no turno)")
    elif camera_ok and ja_avisado:
        registrar_evento("camera_recuperada", grupo, f"{total_erros} falhas no turno")


def dados_publicos_job(job):
    """Remove campos internos antes de enviar o estado para o navegador."""
    if not job:
        return {
            "status": "sem_sessao",
            "mensagem": "Aguardando sessão de avaliação...",
            "grupo": "",
            "nome_exibir": "",
            "turma": "",
            "rodada": 0,
            "duracao_s": 0,
            "tempo_restante_s": 0,
            "progresso": 0,
            "frames_processados": 0,
            "tempo_ms": 0,
            "classes": [],
            "classes_frame": [],
            "camera_ok": True,
            "camera_mensagem": "",
            "camera_erros": 0,
            "imagem_url": "",
            "ranking_url": "/ranking",
            "auditoria_url": "/auditoria"
        }
    ocultos = {"camera_url"}
    return {chave: valor for chave, valor in job.items() if not chave.startswith("_") and chave not in ocultos}


def obter_job_ativo_publico():
    """Retorna a sessão ativa do telão, se existir."""
    with AVALIACOES_LOCK:
        job = AVALIACOES_AO_VIVO.get(TELAO_JOB_ATIVO_ID)
        return dados_publicos_job(job)


def obter_job_publico(job_id):
    """Retorna uma sessão específica, sem campos internos."""
    with AVALIACOES_LOCK:
        job = AVALIACOES_AO_VIVO.get(job_id)
        return dados_publicos_job(job) if job else None


def definir_job_ativo(job_id):
    """Define qual sessão deve aparecer na tela do telão."""
    global TELAO_JOB_ATIVO_ID
    with AVALIACOES_LOCK:
        if job_id in AVALIACOES_AO_VIVO:
            TELAO_JOB_ATIVO_ID = job_id
            return True
    return False


def callback_ao_vivo(job_id, classes_esperadas, frame, frames_processados, tempo_ms, estatisticas,
                     classes_frame, tempo_decorrido, duracao_s):
    """Atualiza o telão com o frame anotado, o cronômetro e as classes do desafio."""
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        publicar_frame_telao(jpeg.tobytes())

    progresso = min(100, int((tempo_decorrido / duracao_s) * 100)) if duracao_s else 0
    tempo_restante = max(0, int(duracao_s - tempo_decorrido))
    atualizar_job_ao_vivo(
        job_id,
        status="rodando",
        mensagem="Turno em andamento.",
        frames_processados=frames_processados,
        tempo_ms=round(tempo_ms, 1),
        progresso=progresso,
        tempo_restante_s=tempo_restante,
        classes=resumo_classes_ao_vivo(estatisticas, classes_esperadas),
        classes_frame=classes_frame,
        imagem_url="/telao/stream"
    )


def publicar_frame_telao(jpeg):
    """Troca o frame do telão e acorda quem está assistindo o stream."""
    with FRAME_TELAO_SINAL:
        FRAME_TELAO["jpeg"] = jpeg
        FRAME_TELAO["contador"] += 1
        FRAME_TELAO_SINAL.notify_all()


def executar_job_avaliacao(job_id, grupo, caminho_modelo, camera_url, duracao_s, rodada, pasta_grupo_anotadas):
    """Executa uma avaliação em background para liberar a tela ao vivo."""
    try:
        atualizar_job_ao_vivo(job_id, status="carregando", mensagem="Carregando best.pt...")
        modelo = obter_modelo(caminho_modelo)
        classes_esperadas = ler_classes_oficiais()

        atualizar_job_ao_vivo(job_id, status="rodando", mensagem="Avaliando modelo...")
        avaliacoes_novas, evidencias, frames_processados = avaliar_camera(
            modelo,
            camera_url,
            duracao_s,
            pasta_grupo_anotadas,
            classes_esperadas,
            callback_ao_vivo=lambda *args: callback_ao_vivo(job_id, classes_esperadas, *args),
            ao_status_camera=lambda ok, parado, erros: atualizar_status_camera(job_id, ok, parado, erros)
        )

        salvar_avaliacoes(
            grupo["grupo"],
            caminho_modelo,
            avaliacoes_novas,
            evidencias=evidencias,
            stream_url=camera_url,
            duracao_s=duracao_s,
            frames_processados=frames_processados,
            rodada=rodada
        )

        detectadas = [item for item in avaliacoes_novas if item.get("classe_detectada") != "nenhuma"]
        total_deteccoes = sum(item.get("deteccoes", 0) for item in detectadas)
        atualizar_job_ao_vivo(
            job_id,
            status="concluida",
            mensagem=(f"Tempo esgotado: {len(detectadas)} classe(s), {total_deteccoes} detecções, "
                      f"{len(evidencias)} evidências para auditar."),
            progresso=100,
            tempo_restante_s=0,
            frames_processados=frames_processados,
            ranking_url="/ranking",
            auditoria_url="/auditoria"
        )
    except Exception as erro:
        atualizar_job_ao_vivo(
            job_id,
            status="erro",
            mensagem=f"Erro ao avaliar a câmera: {erro}",
            progresso=100,
            tempo_restante_s=0
        )


def preparar_job_avaliacao(grupo, caminho_modelo, camera_url, duracao_s, rodada, pasta_grupo_anotadas):
    """Prepara uma avaliação e deixa a tela do telão aguardando START."""
    global TELAO_JOB_ATIVO_ID
    job_id = uuid.uuid4().hex
    with AVALIACOES_LOCK:
        AVALIACOES_AO_VIVO[job_id] = {
            "id": job_id,
            "status": "preparada",
            "mensagem": "Sessão pronta. Clique em START",
            "grupo": grupo["grupo"],
            "nome_exibir": grupo.get("nome_exibir") or grupo["grupo"],
            "turma": grupo.get("turma", ""),
            "rodada": rodada,
            "duracao_s": duracao_s,
            "tempo_restante_s": duracao_s,
            "progresso": 0,
            "frames_processados": 0,
            "tempo_ms": 0,
            "classes": resumo_classes_ao_vivo({}, ler_classes_oficiais()),
            "classes_frame": [],
            "camera_ok": True,
            "camera_mensagem": "",
            "camera_erros": 0,
            "imagem_url": "",
            "ranking_url": "",
            "auditoria_url": "",
            "_grupo": grupo,
            "_caminho_modelo": caminho_modelo,
            "_camera_url": camera_url,
            "_pasta_anotadas": pasta_grupo_anotadas,
            "_thread_iniciada": False
        }
        TELAO_JOB_ATIVO_ID = job_id

    # o telão não pode mostrar o último frame do grupo anterior
    publicar_frame_telao(None)

    # o modelo comeca a carregar agora, enquanto o professor ainda arruma o
    # participante: quando o START for apertado, ele ja esta quente
    preaquecer_modelo(caminho_modelo)
    return job_id


def iniciar_job_preparado(job_id):
    """Inicia uma sessão preparada, se ela ainda não estiver rodando."""
    with AVALIACOES_LOCK:
        job = AVALIACOES_AO_VIVO.get(job_id)
        if not job:
            return False, "Sessão não encontrada."
        if job.get("status") in {"carregando", "rodando"}:
            return True, "Sessão já está em andamento."
        if job.get("status") == "concluida":
            return False, "Esta sessão já foi concluída."
        if job.get("status") != "preparada":
            return False, "Esta sessão não está pronta para iniciar."

        job["_thread_iniciada"] = True
        job["status"] = "carregando"
        job["mensagem"] = "Carregando best.pt..."
        grupo = job["_grupo"]
        caminho_modelo = job["_caminho_modelo"]
        camera_url = job["_camera_url"]
        duracao_s = job["duracao_s"]
        rodada = job["rodada"]
        pasta_grupo_anotadas = job["_pasta_anotadas"]

    thread = threading.Thread(
        target=executar_job_avaliacao,
        args=(job_id, grupo, caminho_modelo, camera_url, duracao_s, rodada, pasta_grupo_anotadas),
        daemon=True
    )
    thread.start()
    return True, "Sessão iniciada."


def abrir_janelas_iniciais():
    """Abre a janela principal e a tela do telão quando o app inicia."""
    webbrowser.open(f"http://127.0.0.1:{APP_PORT}")
    webbrowser.open(f"http://127.0.0.1:{APP_PORT}/telao")


iniciar_banco()
migrar_json_para_sqlite()


# ------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------

def destino_seguro(destino):
    """Aceita só caminhos internos do app no ?next= (evita redirecionar para fora)."""
    if destino and destino.startswith("/") and not destino.startswith("//") and "\\" not in destino:
        return destino
    return url_for("index")


@app.before_request
def exigir_login():
    """Todas as telas pedem login, menos a própria tela de login e o CSS/logo."""
    if not LOGIN_ATIVO or session.get("logado"):
        return None
    if request.endpoint in ("login", "sessao", "static"):
        return None

    # Navegação do navegador vai para o login; chamadas fetch/stream recebem
    # 401, para a tela não tentar ler a página de login como JSON.
    if "text/html" in request.headers.get("Accept", ""):
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
    return jsonify({"ok": False, "erro": "Sessão expirada. Faça login de novo."}), 401


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login único do professor (usuário e hash da senha vêm do .env)."""
    if not LOGIN_ATIVO:
        return redirect(url_for("index"))

    destino = destino_seguro(request.values.get("next", ""))

    if request.method == "POST":
        usuario = request.form.get("usuario", "")
        senha = request.form.get("senha", "")
        usuario_ok = hmac.compare_digest(usuario.encode(), ADMIN_USER.encode())
        senha_ok = check_password_hash(ADMIN_PASSWORD_HASH, senha)
        if usuario_ok and senha_ok:
            session.clear()
            session.permanent = True
            session["logado"] = True
            return redirect(destino)
        time.sleep(1)  # freia tentativas em sequência
        flash("Usuário ou senha incorretos.", "erro")

    return render_template("login.html", next=destino)


@app.route("/sessao")
def sessao():
    """A tela de login consulta isto para seguir sozinha quando outra aba entra."""
    return jsonify({"logado": bool(session.get("logado"))})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.errorhandler(413)
def upload_grande_demais(_erro):
    """Upload acima de MAX_UPLOAD_MB."""
    limite = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    mensagem = f"Arquivo grande demais. O limite é {limite} MB."
    if request.path == "/analisar-modelo":
        return jsonify({"ok": False, "erro": mensagem}), 413
    flash(mensagem, "erro")
    return redirect(request.path)


@app.context_processor
def injetar_desafio():
    """Nome e subtítulo do evento em todas as telas (white label)."""
    return {"desafio": ler_desafio()}


@app.route("/analisar-modelo", methods=["POST"])
def analisar_modelo():
    """Carrega temporariamente um best.pt e devolve as classes detectadas."""
    arquivo_pt = request.files.get("modelo")
    if not arquivo_pt or arquivo_pt.filename == "":
        return jsonify({"ok": False, "erro": "Envie um arquivo best.pt."}), 400

    if not arquivo_pt.filename.lower().endswith(".pt"):
        return jsonify({"ok": False, "erro": "O arquivo precisa ter extensão .pt."}), 400

    temporario = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    caminho_temporario = temporario.name
    temporario.close()

    try:
        arquivo_pt.save(caminho_temporario)
        resumo = resumir_modelo(caminho_temporario)
        return jsonify({"ok": True, **resumo})
    except Exception as erro:
        return jsonify({
            "ok": False,
            "erro": f"Não foi possível carregar o modelo com Ultralytics: {erro}"
        }), 400
    finally:
        if os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)


@app.route("/validar-modelo", methods=["GET", "POST"])
def validar_modelo():
    """Tela de sala de aula: confere um best.pt contra as classes oficiais.

    Não cadastra grupo nem guarda o arquivo. Serve para o professor mostrar
    ao aluno, antes da entrega, se as classes batem com o desafio.
    """
    resultado = None
    nome_arquivo = ""

    if request.method == "POST":
        arquivo_pt = request.files.get("modelo")

        if not arquivo_pt or arquivo_pt.filename == "":
            flash("Selecione um arquivo best.pt.", "erro")
            return redirect(url_for("validar_modelo"))

        if not arquivo_pt.filename.lower().endswith(".pt"):
            flash("O arquivo precisa ter extensão .pt.", "erro")
            return redirect(url_for("validar_modelo"))

        nome_arquivo = os.path.basename(arquivo_pt.filename)
        temporario = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        caminho_temporario = temporario.name
        temporario.close()

        try:
            arquivo_pt.save(caminho_temporario)
            resultado = resumir_modelo(caminho_temporario)
        except Exception as erro:
            flash(f"Não foi possível carregar o modelo com Ultralytics: {erro}", "erro")
            return redirect(url_for("validar_modelo"))
        finally:
            if os.path.exists(caminho_temporario):
                os.remove(caminho_temporario)

    return render_template("validar_modelo.html",
                           oficiais=classes_oficiais_detalhadas(),
                           resultado=resultado,
                           nome_arquivo=nome_arquivo)


@app.route("/", methods=["GET", "POST"])
def index():
    """Cadastro de grupo e upload do best.pt."""
    dados = ler_resultados()

    if request.method == "POST":
        nome_grupo  = request.form.get("nome_grupo", "").strip()
        turma       = request.form.get("turma", "").strip()
        integrantes = request.form.get("integrantes", "").strip()
        arquivo_pt  = request.files.get("modelo")

        if not nome_grupo:
            flash("Informe o nome do grupo.", "erro")
            return redirect(url_for("index"))

        if not arquivo_pt or arquivo_pt.filename == "":
            flash("Faça o upload do arquivo best.pt.", "erro")
            return redirect(url_for("index"))

        # normaliza o nome para usar como pasta
        nome_norm = normalizar_nome(nome_grupo)
        pasta_grupo = os.path.join(PASTA_MODELOS, nome_norm)
        os.makedirs(pasta_grupo, exist_ok=True)

        caminho_modelo = os.path.join(pasta_grupo, "best.pt")

        # grava primeiro em um arquivo temporário: um envio recusado
        # nunca pode apagar o último modelo já aprovado do grupo
        # o Ultralytics só carrega arquivos terminados em .pt
        caminho_novo = os.path.join(pasta_grupo, "best_novo.pt")
        arquivo_pt.save(caminho_novo)

        def recusar(mensagem):
            if os.path.exists(caminho_novo):
                os.remove(caminho_novo)
            flash(mensagem, "erro")
            return redirect(url_for("index"))

        try:
            resumo_modelo = resumir_modelo(caminho_novo)
        except Exception as e:
            return recusar(f"Não foi possível carregar o best.pt com Ultralytics: {e}")

        if resumo_modelo["total_classes"] == 0:
            return recusar("O modelo carregou, mas não informou classes. Envie outro best.pt.")

        if resumo_modelo["parece_coco"]:
            return recusar("O modelo parece ser COCO padrão. Envie o best.pt treinado pelo grupo.")

        validacao = resumo_modelo["validacao"]
        if validacao["status"] == "reprovado":
            return recusar(f"Modelo reprovado na conferência das classes oficiais. {validacao['mensagem']}")

        # só agora substitui o modelo anterior
        os.replace(caminho_novo, caminho_modelo)

        if validacao["status"] == "aprovado_com_aviso":
            flash(validacao["mensagem"], "ok")

        grupo_salvo = {
            "data_hora":   datetime.now().isoformat(),
            "grupo":       nome_norm,
            "nome_exibir": nome_grupo,
            "turma":       turma,
            "integrantes": integrantes,
            "modelo":      caminho_modelo,
            "avaliacoes":  []
        }

        if buscar_grupo(dados, nome_norm):
            salvar_grupo(grupo_salvo)
            registrar_evento("upload_modelo", nome_norm, f"Modelo atualizado: {caminho_modelo}")
            flash(f"Modelo atualizado para o grupo '{nome_grupo}' com {resumo_modelo['total_classes']} classes.", "ok")
        else:
            salvar_grupo(grupo_salvo)
            registrar_evento("cadastro_grupo", nome_norm, f"Modelo recebido: {caminho_modelo}")
            flash(f"Grupo '{nome_grupo}' cadastrado com sucesso! Modelo com {resumo_modelo['total_classes']} classes.", "ok")

        return redirect(url_for("index"))

    return render_template("index.html", dados=dados)


@app.route("/avaliar", methods=["GET", "POST"])
def avaliar():
    """Avaliação do modelo de um grupo usando /capture do ESP32-CAM."""
    dados = ler_resultados()

    if request.method == "POST":
        nome_grupo = request.form.get("grupo", "").strip()
        camera_url = request.form.get("camera_url", "").strip()
        duracao_s = ler_desafio()["duracao_s"]

        if not ler_classes_oficiais():
            flash("Cadastre as classes do desafio antes de avaliar.", "erro")
            return redirect(url_for("desafio"))

        if not nome_grupo:
            flash("Selecione um grupo.", "erro")
            return redirect(url_for("avaliar"))

        if not camera_url:
            flash("Informe a URL do ESP32-CAM.", "erro")
            return redirect(url_for("avaliar"))

        camera_url = normalizar_camera_url(camera_url)

        grupo = buscar_grupo(dados, nome_grupo)
        if not grupo:
            flash("Grupo não encontrado.", "erro")
            return redirect(url_for("avaliar"))

        caminho_modelo = grupo["modelo"]
        if not os.path.exists(caminho_modelo):
            flash(f"Arquivo best.pt não encontrado: {caminho_modelo}", "erro")
            return redirect(url_for("avaliar"))

        rodada = proxima_rodada(nome_grupo)
        pasta_grupo_anotadas = os.path.join(PASTA_ANOTADAS, nome_grupo, f"rodada_{rodada:03d}")

        job_id = preparar_job_avaliacao(
            grupo,
            caminho_modelo,
            camera_url,
            duracao_s,
            rodada,
            pasta_grupo_anotadas
        )
        flash("Sessão preparada. Clique em START no painel do Turno.", "ok")
        return redirect(url_for("arena_admin"))

    return render_template("avaliar.html",
                           dados=dados,
                           abrir_telao=request.args.get("abrir_telao") == "1",
                           job_ativo=obter_job_ativo_publico())


@app.route("/ao-vivo/<job_id>")
def ao_vivo(job_id):
    """Compatibilidade: ativa uma sessão específica e abre o telão."""
    if not definir_job_ativo(job_id):
        flash("Avaliação ao vivo não encontrada. Inicie uma nova avaliação.", "erro")
        return redirect(url_for("avaliar"))
    return redirect(url_for("telao"))


@app.route("/telao")
def telao():
    """Tela fixa do telão, sem menu, aguardando ou exibindo a sessão ativa."""
    return render_template("ao_vivo.html", confidence_minima=CONFIDENCE_MINIMA)


@app.route("/telao/status")
def status_telao():
    """Retorna a sessão atualmente selecionada para o telão."""
    return jsonify(obter_job_ativo_publico())


@app.route("/telao/stream")
def stream_telao():
    """MJPEG com o frame anotado mais recente: um quadro novo assim que fica pronto."""
    def gerar():
        visto = -1
        while True:
            with FRAME_TELAO_SINAL:
                # o timeout reenvia o mesmo quadro de vez em quando, o que
                # também derruba o laço quando o navegador fecha a conexão
                FRAME_TELAO_SINAL.wait_for(lambda: FRAME_TELAO["contador"] != visto, timeout=5)
                jpeg = FRAME_TELAO["jpeg"]
                visto = FRAME_TELAO["contador"]
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(gerar(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telao/start", methods=["POST"])
def start_telao():
    """Inicia a sessão preparada que está ativa no telão."""
    with AVALIACOES_LOCK:
        job_id = TELAO_JOB_ATIVO_ID
    if not job_id:
        return jsonify({"ok": False, "mensagem": "Nenhuma sessão preparada."}), 400

    ok, mensagem = iniciar_job_preparado(job_id)
    status_code = 200 if ok else 400
    return jsonify({"ok": ok, "mensagem": mensagem}), status_code


@app.route("/arena/reset", methods=["POST"])
def arena_reset():
    """Limpa os dados de ensaio antes do evento, sempre com backup antes."""
    escopo = request.form.get("escopo", "resultados")
    confirmacao = request.form.get("confirmacao", "").strip().upper()

    if escopo not in {"resultados", "tudo"}:
        flash("Escopo inválido para a limpeza.", "erro")
        return redirect(url_for("arena_admin"))

    if confirmacao != "LIMPAR":
        flash("Digite LIMPAR no campo de confirmação para executar a limpeza.", "erro")
        return redirect(url_for("arena_admin"))

    with AVALIACOES_LOCK:
        rodando = [job for job in AVALIACOES_AO_VIVO.values()
                   if job.get("status") in {"carregando", "rodando"}]
    if rodando:
        flash("Há um turno em andamento. Encerre antes de limpar a competição.", "erro")
        return redirect(url_for("arena_admin"))

    backup, contagens = limpar_competicao(escopo)
    apagados = ", ".join(f"{quantidade} {tabela}" for tabela, quantidade in contagens.items())
    flash(f"Competição limpa ({escopo}): {apagados}. Backup salvo em {backup}", "ok")
    return redirect(url_for("arena_admin"))


@app.route("/arena-admin")
def arena_admin():
    """Painel do turno: START, cronômetro e classes detectadas."""
    return render_template(
        "arena_admin.html",
        job_ativo=obter_job_ativo_publico(),
        confidence_minima=CONFIDENCE_MINIMA,
        nome_dispositivo=NOME_DISPOSITIVO,
        usando_gpu=(DISPOSITIVO == "cuda")
    )


@app.route("/desafio", methods=["GET", "POST"])
def desafio():
    """Cadastro do desafio: nome do evento, classes esperadas e duração do turno."""
    if request.method == "POST":
        nome_evento = request.form.get("nome_evento", "").strip()
        subtitulo = request.form.get("subtitulo", "").strip()
        classes = normalizar_classes(request.form.get("classes", ""))
        try:
            duracao_s = int(request.form.get("duracao_s", ""))
        except ValueError:
            duracao_s = 0

        if not nome_evento:
            flash("Informe o nome do evento.", "erro")
        elif not classes:
            flash("Informe ao menos uma classe.", "erro")
        elif not 10 <= duracao_s <= 600:
            flash("A duração do turno precisa ficar entre 10 e 600 segundos.", "erro")
        else:
            salvar_desafio(nome_evento, subtitulo, duracao_s, classes)
            registrar_evento("desafio", "", f"{nome_evento}: {', '.join(classes)} ({duracao_s}s)")
            flash(f"Desafio salvo com {len(classes)} classes.", "ok")
        return redirect(url_for("desafio"))

    return render_template("desafio.html",
                           classes=ler_classes_oficiais(),
                           arquivo_oficial=ARQUIVO_CLASSES_OFICIAIS)


@app.route("/desafio/classes-oficiais.txt")
def baixar_classes_oficiais():
    """Baixa o próprio classes-oficiais.txt, o mesmo arquivo que o app usa."""
    if not os.path.exists(ARQUIVO_CLASSES_OFICIAIS):
        flash("Cadastre as classes antes de baixar o arquivo.", "erro")
        return redirect(url_for("desafio"))
    return send_from_directory(os.path.dirname(ARQUIVO_CLASSES_OFICIAIS),
                               os.path.basename(ARQUIVO_CLASSES_OFICIAIS),
                               as_attachment=True, mimetype="text/plain")


@app.route("/ao-vivo/<job_id>/status")
def status_ao_vivo(job_id):
    """Retorna o estado atual da avaliação ao vivo."""
    job = obter_job_publico(job_id)
    if not job:
        return jsonify({"status": "erro", "mensagem": "Avaliação não encontrada."}), 404
    return jsonify(job)


@app.route("/ranking")
def ranking():
    """Classificação dos grupos avaliados."""
    tabela_rank = calcular_ranking(ler_resultados())
    return render_template("ranking.html",
                           ranking=tabela_rank,
                           classes_esperadas=ler_classes_oficiais(),
                           pendentes_total=sum(item["evidencias_pendentes"] for item in tabela_rank))


@app.route("/criterios")
def criterios():
    """Explica como as evidências validadas entram no cálculo."""
    return render_template(
        "criterios.html",
        confidence_minima=CONFIDENCE_MINIMA,
        top_predicoes=TOP_PREDICOES_POR_CLASSE,
        intervalo_evidencia=INTERVALO_EVIDENCIA_POR_CLASSE
    )


@app.route("/dashboard")
def dashboard():
    """Pódio do resultado final, revelado pelo professor."""
    ranking = calcular_ranking(ler_resultados())
    top3 = [item for item in ranking if item["tem_pontuacao"]][:3]

    grupo_selecionado = request.args.get("grupo", "").strip()
    grupos_ranking = {item["grupo"] for item in ranking}
    if grupo_selecionado not in grupos_ranking:
        grupo_selecionado = ranking[0]["grupo"] if ranking else ""

    evidencias_grupo = listar_melhores_evidencias_por_classe(grupo_selecionado) if grupo_selecionado else []

    return render_template("dashboard.html",
                           ranking=ranking,
                           top3=top3,
                           grupo_selecionado=grupo_selecionado,
                           evidencias_grupo=evidencias_grupo,
                           pendentes_total=sum(item["evidencias_pendentes"] for item in ranking))


@app.route("/auditoria")
def auditoria():
    """Exibe histórico de avaliações e eventos do app."""
    return render_template("auditoria.html",
                           historico=listar_historico(),
                           eventos=listar_eventos(),
                           evidencias=listar_evidencias_agrupadas())


@app.route("/evidencia/<int:evidencia_id>/<status>", methods=["POST"])
def revisar_evidencia(evidencia_id, status):
    """Marca uma evidência visual como validada, rejeitada ou pendente."""
    atualizar_status_evidencia(evidencia_id, status)
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify({"id": evidencia_id, "status": status})
    return redirect(f"{url_for('auditoria')}#evidencia-{evidencia_id}")


@app.route("/imagens_anotadas/<path:caminho>")
def servir_imagem_anotada(caminho):
    """Serve as imagens anotadas salvas."""
    return send_from_directory(PASTA_ANOTADAS, caminho)


def gerar_hash_senha():
    """python app.py --senha : pede a senha e imprime a linha para o .env."""
    import getpass
    senha = getpass.getpass("Nova senha do professor: ")
    if senha != getpass.getpass("Repita a senha: "):
        print("As senhas não conferem.")
        return
    # aspas simples: o hash tem '$' e o .env não deve interpretá-lo
    print("\nCopie para o .env:")
    print(f"ADMIN_PASSWORD_HASH='{generate_password_hash(senha)}'")


if __name__ == "__main__":
    if "--senha" in sys.argv:
        gerar_hash_senha()
        sys.exit(0)

    print("Iniciando app do professor...")
    print(f"Acesse: http://127.0.0.1:{APP_PORT}")
    if APP_HOST != "127.0.0.1":
        print(f"[seguranca] AVISO: app aberto na rede em {APP_HOST}:{APP_PORT}.")
    if not LOGIN_ATIVO:
        print("[seguranca] AVISO: login desligado (sem ADMIN_PASSWORD_HASH no .env). "
              "Gere com: python app.py --senha")
    if ABRIR_JANELAS_AO_INICIAR:
        threading.Timer(1.2, abrir_janelas_iniciais).start()
    app.run(debug=False, host=APP_HOST, port=APP_PORT, use_reloader=False)
