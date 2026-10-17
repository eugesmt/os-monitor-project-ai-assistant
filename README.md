# Standalone Developer Assistant

Это отдельный проект для задания «День 31. Ассистент разработчика». Он
подключается к соседнему Task Manager AI как к исходному проекту: сам Task
Manager не модифицируется.

## Что реализовано

```text
README.md + docs/**/*.md + Go source (+ TASKMANAGER_MCP.md)
        ↓ structure-aware documentation/code chunks
qwen3-embedding:0.6b через Ollama
        ↓
FAISS + SQLite (локальные generated artifacts)
        ↓ top-k retrieval
/help <вопрос> → DeepSeek-compatible API
        ↓
MCP stdio client → MCP server → get_git_branch(project root)
```

Есть два независимых локальных сценария:

- `/help` отвечает на вопрос по документации и коду, получает текущую ветку
  через MCP `get_git_branch` и показывает источники. В этом MCP-пайплайне CLI
  напрямую Git не вызывает.
- `review` сам безопасно вызывает `git diff`, добавляет найденный RAG-контекст
  и просит DeepSeek проверить изменение.

MCP-сервер принимает только заранее заданный `--project-root`, а tool не
принимает путь или shell-команду от пользователя.

## Установка и запуск

Нужен установленный [uv](https://docs.astral.sh/uv/). Из корня standalone-проекта
достаточно положить ключ в его собственный `.env` и запустить команду:

```bash
cd /home/lab-dev/Projects/taskmanager-ai-assistant
uv run taskmanager-assistant
```

По умолчанию ассистент ищет целевой проект в соседнем каталоге
`../taskmanager-ai`. Для другого расположения задайте
`TASKMANAGER_PROJECT_ROOT` или передайте `--project-root`. Ассистент
автоматически читает только `/home/lab-dev/Projects/taskmanager-ai-assistant/.env`;
`.env` целевого `taskmanager-ai` не читается. Уже экспортированные переменные
имеют приоритет. Если нужно явно загрузить файл через `uv`, это также поддерживается:

```bash
uv run --env-file .env taskmanager-assistant
```

`.env` уже добавлен в `.gitignore`; для шаблона без секретов используйте
`.env.example`.

Нужны запущенный Ollama и модель:

```bash
ollama pull qwen3-embedding:0.6b
```

Генерация выполняется только через внешний DeepSeek API (`DEEPSEEK_MODEL`, по
умолчанию `deepseek-chat`): туда отправляются сам diff и найденный контекст
исходников/документации. Qwen используется только для embeddings через
локальный Ollama; FAISS/SQLite и embeddings остаются локальными. Имена всех
настроек находятся в `.env.example`; секреты туда не добавляются.

## Индексация и запуск

При первом запуске или при обнаружении устаревшего индекса команда автоматически
строит FAISS + SQLite заново, а затем открывает интерактивный чат или запускает
review. Это может занять время из-за локальных embedding-запросов. Повторный
запуск использует готовый индекс. Для принудительной перестройки добавьте
`--reindex`:

```bash
uv run taskmanager-assistant --reindex
```

После запуска введите `/help <вопрос>`. Расширенные подкоманды по-прежнему
доступны:

```bash
uv run taskmanager-assistant build-index \
  --project-root /home/lab-dev/Projects/taskmanager-ai

uv run taskmanager-assistant help \
  --project-root /home/lab-dev/Projects/taskmanager-ai \
  'Где находится composition root проекта?'
```

Команда `help` печатает ответ DeepSeek, MCP branch и использованные источники.
Индексируются документация (`README.md`, `docs/**/*.md` и существующий
`TASKMANAGER_MCP.md`) и исходный Go-код проекта; generated
JSONL/FAISS/SQLite не коммитятся.

В интерактивном режиме индекс строится лениво: запуск без команды не требует
Ollama, а индекс проверяется только при первом `/help`. Для файлового агента
доступны две slash-команды:

```text
/files Найди все использования POST /api/ai/ask и обнови документацию
/files-read Найди все использования POST /api/ai/ask
```

`/files` может изменять файлы согласно цели и показывает итоговый diff.
`/files-read` предназначена для поиска и анализа: write-инструменты MCP
технически отключены. Обычный CLI-вызов имеет тот же режим через
`files --read-only`.

## Файловый ассистент

Команда `files` принимает цель на естественном языке, сама находит нужные
файлы и выполняет разрешённые операции через MCP. Пользователю не нужно
заранее указывать имя файла. Агент возвращает итог, список прочитанных и
изменённых файлов и итоговый `git diff`. Запуск не требует индексации RAG;
единственный внешний вызов при обычной работе — DeepSeek для выбора шагов.

Найти использования API в проекте (read-only цель):

```bash
uv run taskmanager-assistant files \
  --project-root /home/lab-dev/Projects/taskmanager-ai \
  "Найди все использования POST /api/ai/ask и укажи вызывающие файлы"
```

Обновить документацию после изучения реализации:

```bash
uv run taskmanager-assistant files \
  --project-root /home/lab-dev/Projects/taskmanager-ai \
  "Проверь README и документацию API; добавь описание endpoint POST /api/ai/ask, если его не хватает"
```

Для каждой цели агент ограничен числом итераций и вызовов инструментов.
Чтение, поиск, запись, патч и получение diff выполняются только через
allow-list MCP; shell-команды, произвольные пути и subprocess модели
недоступны. Если цель не требует изменений, отчёт содержит `Изменённые
файлы: нет`.

## Локальное code review

До подключения GitHub Actions review можно запускать локально из корня
standalone-проекта. Команда сама проверяет актуальность RAG-индекса, передаёт
diff и связанный код/документацию в DeepSeek и печатает отчёт:

```bash
make review
```

По умолчанию сравниваются `develop...HEAD` соседнего `taskmanager-ai`. Ref-ы и
путь к проекту можно переопределить:

```bash
make review BASE=develop HEAD=feature/my-change PROJECT_ROOT=../taskmanager-ai
```

Для незакоммиченных изменений рабочего дерева используйте:

```bash
make review-working-tree
```

Те же режимы доступны напрямую через CLI. Для сохранения отчёта добавьте
`--output review.md`:

```bash
uv run taskmanager-assistant review \
  --project-root ../taskmanager-ai \
  --base develop --head HEAD --output review.md

uv run taskmanager-assistant review \
  --project-root ../taskmanager-ai --working-tree
```

Режимы `--working-tree` и `--base/--head` взаимоисключающие.
Каждая часть ревью возвращает ровно три раздела: `Потенциальные баги`,
`Архитектурные проблемы` и `Рекомендации`; конкретные замечания содержат
файл, строку, серьёзность и объяснение. Если изменений нет, печатается
короткое сообщение.

Большой diff разбивается на части размером до `REVIEW_MAX_DIFF_CHARS` (по
умолчанию `60000`) и отправляется несколькими последовательными вызовами
DeepSeek без дополнительного synthesis-запроса. Поэтому такой review может
работать заметно дольше и стоить дороже. В TTY строка
`Reviewing batch N/M...` обновляется на месте, а в CI/non-TTY каждая часть
печатается отдельной строкой.

Для безопасного точечного ревью указывайте ближайшую базу, например
`--base HEAD~1 --head HEAD`, либо актуальную ветку проекта `develop`. В режиме
`--working-tree` проверяются staged и unstaged изменения; новые `untracked`
файлы сначала добавьте через `git add`, иначе `git diff HEAD` их не увидит.

GitHub Action и автоматическая публикация комментариев в PR пока не реализованы;
текущий review запускается только локально. Встроенного inline-review также
нет.

## Проверка

```bash
python3 -m unittest discover -s tests -v
```

Тест MCP запускает настоящий stdio-сервер, выполняет `initialize`, `tools/list`
и `tools/call`, после чего проверяет ветку реального временного Git-репозитория.
Для smoke test с Ollama сначала выполните `build-index`, затем `help`.

## Поддерживаемый MCP subset

Сервер реализует line-delimited JSON-RPC с методами `initialize`, `tools/list`
и `tools/call`; уведомление `notifications/initialized` принимается без ответа.
Для задания достаточно этого subset, который используется встроенным MCP
клиентом. Дополнительные MCP resources/prompts не нужны.
