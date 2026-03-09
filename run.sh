#!/bin/bash
set -e
docker compose -f deploy/docker-compose.yml up --build
