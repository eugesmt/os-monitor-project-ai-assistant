PROJECT_ROOT ?= ../taskmanager-ai
BASE ?= develop
HEAD ?= HEAD

.PHONY: review review-working-tree

review:
	uv run taskmanager-assistant review \
		--project-root "$(PROJECT_ROOT)" \
		--base "$(BASE)" \
		--head "$(HEAD)"

review-working-tree:
	uv run taskmanager-assistant review \
		--project-root "$(PROJECT_ROOT)" \
		--working-tree
