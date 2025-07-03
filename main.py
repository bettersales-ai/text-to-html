import re

import modal
from transformers import AutoTokenizer, AutoModelForCausalLM


image = modal.Image.debian_slim().pip_install_from_pyproject(
    "./pyproject.toml",
)

app = modal.App(image=image, name="example-lifecycle-web")

# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(
        SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html


def create_prompt(
    text: str, tokenizer=None, instruction: str = None, schema: str = None
) -> str:
    """
    Create a prompt for the model with optional instruction and JSON schema.
    """
    if not instruction:
        instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    if schema:
        instruction = "Extract the specified information from a list of news threads and present it in a structured JSON format."
        prompt = f"{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```"
    else:
        prompt = f"{instruction}\n```html\n{text}\n```"

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def setup_model():
    device = "cuda"  # or "cpu"
    tokenizer = AutoTokenizer.from_pretrained("jinaai/ReaderLM-v2")
    model = AutoModelForCausalLM.from_pretrained("jinaai/ReaderLM-v2").to(device)

    return tokenizer, model, device


def extract_html(url: str) -> str:
    import cloudscraper

    scraper = cloudscraper.create_scraper()
    """
    Extract HTML content from a given URL using cloudscraper.
    """
    try:
        response = scraper.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise ValueError(f"Failed to fetch HTML from {url}: {e}")


@app.function(gpu="A100")
def hello_world(html: str) -> str:
    html = clean_html(html)

    tokenizer, model, device = setup_model()

    input_prompt = create_prompt(html, tokenizer=tokenizer)
    inputs = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        temperature=0,
        do_sample=False,
        max_new_tokens=1024,
        repetition_penalty=1.08,
    )

    return tokenizer.decode(outputs[0])


@app.function()
@modal.fastapi_endpoint(docs=True)
def hello(url: str) -> str:
    html = extract_html(url)
    if not html:
        return "No HTML content found at the provided URL."
    return hello_world.remote(html)


@app.local_entrypoint()
def main():
    url = "https://www.channelstv.com/2018/12/06/bush-to-be-buried-after-four-day-tribute/"
    html = extract_html(url)
    if not html:
        return "No HTML content found at the provided URL."
    res = hello_world.remote(html)
    print(res)
