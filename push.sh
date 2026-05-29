#!/bin/bash
cd /home/oni/.hermes/profiles/eval-harness/workspace/eval-harness

TOKEN=$(grep -oP 'GITHUB_TOKEN=\S+' /home/oni/.hermes/profiles/eval-harness/.env | cut -d= -f2)

git remote remove origin 2>/dev/null
git remote add origin "https://onicarps:${TOKEN}@github.com/onicarps/eval-harness.git"
git branch -M main
git push -u origin main
