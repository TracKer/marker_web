# Makefile

.PHONY: up
up:
	@echo "====> Building docker..."
	docker compose -f docker-compose.yml build
	@echo "====> Running..."
	docker compose -f docker-compose.yml up
