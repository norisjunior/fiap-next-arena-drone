# App de Avaliação — Fluxo do Professor

Aplicação Flask simples para avaliar os modelos dos grupos, mostrar a inferencia ao vivo no telao, salvar evidencias visuais e gerar o ranking.

O app consome imagens do ESP32-CAM pelo endpoint `/capture`. Para cada grupo, ele guarda as 10 melhores predicoes por classe para revisao visual do professor.

Para evitar imagens quase identicas, o app salva candidatos de evidencia com intervalo minimo por classe. Depois, ordena por confidence e mantem as 10 melhores.

---

## Instalacao das dependencias

Antes de criar o ambiente virtual, confira quais versoes do Python estao instaladas:

```
py -0p
```

Se aparecer Python 3.12 na lista, siga para a proxima seção.

Se não aparecer, instale pelo `winget`:

```
winget install Python.Python.3.12
```

Depois de instalar, feche e abra o terminal novamente. Confira:

```
py -3.12 --version
```

Se o `winget` nao funcionar, use o instalador manual:

1. Acesse `https://www.python.org/downloads/release/python-312/`
2. Baixe o instalador do Windows
3. Durante a instalação, marque **Add python.exe to PATH**
4. Feche e abra o terminal novamente
5. Confira com `py -3.12 --version`

Em computadores do laboratório, o `winget` pode ser bloqueado por politica de TI. Nesse caso, use o instalador manual ou chame o professor/monitor.

Abra o terminal na pasta `app-avaliacao/` e execute:

```
py -3.12 -m venv appProfNextEnv
.\appProfNextEnv\Scripts\Activate.ps1
```

Depois de ativar a venv, confira se o Python correto esta sendo usado. O `python --version` deve mostrar Python 3.12. Se aparecer Python 3.14, a venv nao foi criada ou nao foi ativada com a versao correta.
```
python --version
```

Estando no Python 3.12, prossiga:
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Crie o `.env` a partir do exemplo e gere a chave das sessões do Flask (o `.env` não vai para o git):
```
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```
Cole a chave gerada em `FLASK_SECRET_KEY` dentro do `.env`.

Defina a senha do professor (login único que protege todas as telas):
```
python app.py --senha
```
Copie a linha `ADMIN_PASSWORD_HASH='...'` que aparecer para o `.env`. Sem ela o app roda sem login e avisa no console.

### Segurança

- O app escuta só em `127.0.0.1` (`APP_HOST` no `.env`): apenas o próprio notebook abre as telas. O telão abre no notebook e é espelhado no projetor; as câmeras não precisam acessar o app, é ele quem busca o `/capture`.
- Upload de `best.pt` limitado a `MAX_UPLOAD_MB` (padrão 100 MB).
- Antes de carregar um `best.pt`, o app passa por duas barreiras. Primeiro lê o pickle interno **sem executá-lo** e recusa o arquivo se ele tentar usar algo fora de `torch`/`ultralytics`/tipos básicos (por exemplo `os.system`, `eval`, `exec`). Depois abre o arquivo com `torch.load(weights_only=True)`, que só aceita as classes de `torch`/`ultralytics`: um arquivo que engane a primeira leitura cai aqui. O motivo aparece na tela e no console com o prefixo `[seguranca]`.
- Só entra um YOLO de detecção treinado pelo grupo: o modelo precisa ser `DetectionModel` (YOLO-World e YOLOE são recusados, porque detectam qualquer classe por texto) e trazer os `train_args` de um `model.train()` com `task=detect`. Além disso, as classes precisam bater com a lista oficial e o modelo não pode ser o COCO padrão.

---

## Como iniciar

```
python app.py
```

Acesse no navegador: **http://127.0.0.1:5000**

---

## Páginas da aplicação

| URL | O que faz |
|---|---|
| `/desafio` | Cadastrar nome do evento, classes esperadas e duração do turno; baixar o `classes-oficiais.txt` |
| `/` | Cadastrar grupo e fazer upload do best.pt |
| `/avaliar` | Preparar a sessão de um grupo usando a câmera do ESP32-CAM |
| `/arena-admin` | Turno: START, cronômetro e detecções por classe |
| `/telao` | Tela do projetor com a câmera, a detecção atual, o cronômetro e as classes já detectadas |
| `/ao-vivo/<id>` | Ativar uma sessão específica no telão, mantido por compatibilidade |
| `/ranking` | Ver ranking de todos os grupos |
| `/dashboard` | Ver o podio animado e escolher um grupo para conferir suas melhores deteccoes por classe |
| `/auditoria` | Revisar evidencias por classe e consultar o historico |
| `/criterios` | Ver como validacao, rejeicao e confidence entram no calculo |

---

## Antes de usar

### 1. Cadastre o desafio

Abra `/desafio` e informe o nome do evento, as classes esperadas (uma por linha, na ordem do `data.yaml`) e a duração do turno. As classes são gravadas em `aluno/classes-oficiais.txt` (o mesmo arquivo do autoteste do aluno) quando o app roda dentro do monorepo. Rodando o app sozinho, elas ficam em `dados/classes-oficiais.txt`. Use **Baixar classes-oficiais.txt** para entregar aos grupos.

Cadastre as classes antes de receber os modelos: o cadastro recusa o `best.pt` que não bater com a lista.

### 2. Inicie o ESP32-CAM do professor

Grave o firmware HTTP no ESP32-CAM e confirme a URL:

```
http://IP_DO_ESP32CAM/capture
```

### 3. Receba os best.pt dos grupos

Os arquivos serão salvos automaticamente em `modelos_recebidos/` após o upload.

Ao selecionar um `best.pt` na pagina de cadastro, o app tenta carregar o modelo com Ultralytics antes do envio final. Se carregar, mostra as classes que o modelo consegue detectar. Se parecer um modelo COCO padrao, o cadastro e bloqueado.

### 4. Conduza os turnos

Ao iniciar o software, o app abre a janela principal e a tela `/telao`. A tela do telão pode ficar sempre aberta no projetor.

1. Em `/avaliar`, selecione o grupo, informe a URL `/capture` e clique em **Preparar sessão**.
2. Em `/arena-admin` (Turno), clique em **START**. O cronômetro corre o tempo do desafio e o turno encerra sozinho.
3. Durante o turno, todas as classes do desafio contam. O alvo de cada jogada é decidido fora do app, pelo dado.

Se a câmera cair, o tempo parado é devolvido ao grupo, até 30 segundos.

Depois da avaliação, acesse `/auditoria` para revisar as melhores predicoes por classe.

---

## Estrutura de pastas

```
app-avaliacao/
├── app.py                  ← código principal
├── requirements.txt
├── templates/              ← páginas HTML
│   ├── desafio.html
│   ├── index.html
│   ├── avaliar.html
│   ├── arena_admin.html    ← painel do turno
│   ├── ao_vivo.html        ← telão
│   ├── ranking.html
│   ├── dashboard.html
│   ├── auditoria.html
│   └── criterios.html
├── static/
│   ├── style.css
│   └── LogoFIAP.png
├── modelos_recebidos/      ← best.pt dos grupos (criado automaticamente)
├── imagens_anotadas/       ← evidencias visuais das melhores predicoes
└── dados/
    ├── avaliacao.sqlite    ← banco SQLite com histórico e auditoria
    ├── desafio.json        ← nome do evento, subtítulo e duração do turno
    └── resultados.json     ← arquivo antigo, usado apenas para migração
```

---

## Configuração

No início do `app.py`:

```python
CONFIDENCE_MINIMA = 0.50   # confiança mínima para registrar deteccao
DURACAO_AVALIACAO_PADRAO = 120
INTERVALO_ENTRE_FRAMES = 0.05
TOP_PREDICOES_POR_CLASSE = 5
INTERVALO_EVIDENCIA_POR_CLASSE = 3.0
```

`INTERVALO_ENTRE_FRAMES = 0.05` tenta processar ate 20 frames por segundo. A tela ao vivo usa os proprios frames da avaliacao, sem abrir outra captura em paralelo.

---

## Auditoria

Os cadastros, uploads, URLs usadas, objetos detectados, confidence média e tempos ficam registrados em `dados/avaliacao.sqlite`.

O ranking usa a última rodada de avaliação de cada grupo. As rodadas anteriores não são apagadas e podem ser consultadas na página `/auditoria`.

A auditoria visual agrupa as evidencias por grupo e depois por classe. Abra o grupo, clique no nome da classe e marque cada imagem como validada, rejeitada ou pendente.

O ranking usa somente evidencias validadas. Evidencias pendentes e rejeitadas nao entram no calculo. Se um grupo ainda nao tem nenhuma evidencia validada, ele fica sem pontuacao final.

No historico, `Conf.` e a confidence media da classe naquela rodada. `Min.` e `Max.` mostram a menor e a maior confidence detectada para a mesma classe.

Cada classe do desafio vale a maior confidence validada dela. A pontuação é a média só das classes que o grupo detectou: classe que não saiu no dado não pune o grupo. Como não existe gabarito manual por frame, o ranking não calcula acurácia supervisionada. Ele ordena por:

1. maior pontuação;
2. mais classes pontuadas;
3. mais detecções no turno;
4. menor tempo médio de inferência.
