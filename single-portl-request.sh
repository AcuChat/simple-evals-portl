#!/bin/bash

curl -s -X POST "${PORTL_API_URL}/cascade-chat-completion" \
  -H "Authorization: Bearer ${PORTL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"provider\": \"openai\",
    \"model\": \"gpt-4o\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$1\"}],
    \"temperature\": 0.0
  }" | jq