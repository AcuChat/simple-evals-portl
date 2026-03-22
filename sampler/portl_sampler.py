import time
import os
import requests

from ..types import MessageList, SamplerBase, SamplerResponse
from .. import common


class PortlCompletionSampler(SamplerBase):
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 1200,
        max_retries: int = 5,
    ):
        self.endpoint_url = os.environ.get(
            "PORTL_API_URL", "https://api.portl.ai"
        )+ "/cascade-chat-completion"

        self.api_key = os.environ.get("PORTL_API_KEY")
        if not self.api_key:
            raise RuntimeError("Environment variable PORTL_API_KEY not set")
        self.provider = provider
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.image_format = "base64"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        return {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0

        while True:
            try:
                if not common.has_only_user_assistant_messages(message_list):
                    raise ValueError(
                        f"Portl sampler only supports user and assistant messages, got {message_list}"
                    )

                payload = {
                    "provider": self.provider,
                    "model": self.model,
                    "messages": message_list,
                    "systemPrompt": self.system_message,
                    "temperature": self.temperature,
                }

                headers = {"Authorization": f"Bearer {self.api_key}"}

                if self.max_tokens is not None:
                    payload["max_tokens"] = self.max_tokens

                response = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                
                response.raise_for_status()

                resp_json = response.json()

                response_text = (
                    resp_json.get("response")
                )

                portl_meta = resp_json.get("portlMeta")
                if response_text is None:
                    choices = resp_json.get("choices")
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        response_text = (
                            first.get("message", {}).get("content")
                            or first.get("text")
                        )

                if response_text is None:
                    raise ValueError(
                        f"Endpoint returned JSON without a recognized response field: {resp_json}"
                    )

                actual_queried_message_list: MessageList = []
                if self.system_message:
                    actual_queried_message_list.append(
                        {"role": "system", "content": self.system_message}
                    )
                actual_queried_message_list.extend(message_list)

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={
                        "endpoint_used": self.endpoint_url,
                        "provider": self.provider,
                        "model": self.model,
                        "raw_response": resp_json,
                        "portlMeta": portl_meta,
                    },
                    actual_queried_message_list=actual_queried_message_list,
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if trial >= self.max_retries:
                    raise RuntimeError(
                        f"Portl endpoint failed after {self.max_retries} retries: {e}"
                    ) from e

                backoff = 2 ** trial
                print(
                    f"Connection/timeout error, retry {trial + 1}/{self.max_retries} after {backoff} sec: {e}"
                )
                time.sleep(backoff)
                trial += 1

            except requests.exceptions.HTTPError as e:
                body = None
                try:
                    body = e.response.text
                except Exception:
                    pass
                raise RuntimeError(
                    f"Portl endpoint returned HTTP error: {e}; body={body}"
                ) from e