python - <<'PY'
import os
import sys
import ssl
import json
import asyncio
import importlib.metadata as md
from pathlib import Path

# =========================
# A REMPLACER ICI SEULEMENT
# =========================
ENDPOINT = "https://langchain-platform.prod.twin.cloud.net.intra/lgp/sta-agent-twin-XXXX"
ASSISTANT_ID = "1e1dfe4f-ed6d-5477-b617-a6a9b4089d17"
CA_BUNDLE = "/Users/f39596/Desktop/ACE - MCP/ace/ace/cert/BNPP_bundle.pem"

TEST_MESSAGE = "donne moi des incidents vpn resolus"
# =========================


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_error(e):
    print("ERROR TYPE:", type(e).__module__ + "." + type(e).__name__)
    print("ERROR REPR:", repr(e))

    for attr in ("status_code", "message"):
        if hasattr(e, attr):
            print(f"{attr}:", getattr(e, attr))

    response = getattr(e, "response", None)
    if response is not None:
        print("response.status_code:", getattr(response, "status_code", None))
        text = getattr(response, "text", "")
        if text:
            print("response.text preview:", text[:1500])


def compact(obj, limit=2500):
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = repr(obj)
    return text[:limit] + ("\n...TRUNCATED..." if len(text) > limit else "")


async def main():
    ENDPOINT_CLEAN = ENDPOINT.strip().rstrip("/")
    CA_PATH = Path(CA_BUNDLE).expanduser().resolve()

    # Important : on désactive le tracing pour éviter le bruit LangSmith multipart ingest
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"

    # Important : variables SSL avant import/utilisation SDK
    os.environ["SSL_CERT_FILE"] = str(CA_PATH)
    os.environ["REQUESTS_CA_BUNDLE"] = str(CA_PATH)
    os.environ["CURL_CA_BUNDLE"] = str(CA_PATH)

    section("0. CONFIG")
    print("ENDPOINT =", ENDPOINT_CLEAN)
    print("ASSISTANT_ID =", ASSISTANT_ID)
    print("CA_BUNDLE =", CA_PATH)
    print("CA exists =", CA_PATH.is_file())
    print("SSL_CERT_FILE =", os.environ.get("SSL_CERT_FILE"))
    print("REQUESTS_CA_BUNDLE =", os.environ.get("REQUESTS_CA_BUNDLE"))

    section("1. TEST CERTIFICAT LOCAL")
    if not CA_PATH.is_file():
        print("❌ Le fichier CA_BUNDLE n'existe pas.")
        sys.exit(2)

    try:
        ctx = ssl.create_default_context(cafile=str(CA_PATH))
        certs = ctx.get_ca_certs()
        print("✅ Le fichier PEM est lisible par Python SSL.")
        print("Nombre de certificats chargés =", len(certs))
        print("Default verify paths =", ssl.get_default_verify_paths())
    except Exception as e:
        print("❌ Le fichier existe mais Python SSL ne peut pas le charger.")
        show_error(e)
        sys.exit(2)

    section("2. TEST SSL BRUT AVEC HTTPX")
    try:
        import httpx
        print("httpx version =", md.version("httpx"))
    except Exception as e:
        print("❌ httpx non disponible.")
        show_error(e)
        sys.exit(2)

    async with httpx.AsyncClient(
        verify=str(CA_PATH),
        timeout=httpx.Timeout(20.0, connect=10.0),
        follow_redirects=False,
    ) as http:
        for url in [ENDPOINT_CLEAN, ENDPOINT_CLEAN + "/info"]:
            print("\nGET", url)
            try:
                r = await http.get(url)
                print("status =", r.status_code)
                print("location =", r.headers.get("location"))
                print("content-type =", r.headers.get("content-type"))
                print("body preview =", r.text[:500].replace("\n", "\\n"))
            except Exception as e:
                print("❌ SSL ou réseau KO sur", url)
                show_error(e)

    section("3. TEST SDK LANGGRAPH")
    try:
        print("langgraph_sdk version =", md.version("langgraph-sdk"))
    except Exception:
        print("langgraph_sdk version = inconnue")

    try:
        print("langgraph version =", md.version("langgraph"))
    except Exception:
        print("langgraph version = inconnue")

    try:
        from langgraph_sdk import get_client
        from langgraph.pregel.remote import RemoteGraph

        client = get_client(
            url=ENDPOINT_CLEAN,
            api_key=None,  # important : pas d'API key
        )

        print("✅ get_client OK")
        print("client =", type(client))
    except Exception as e:
        print("❌ Impossible de créer le client LangGraph.")
        show_error(e)
        sys.exit(3)

    section("4. TEST ASSISTANT")
    try:
        assistant = await client.assistants.get(ASSISTANT_ID)
        print("✅ assistant trouvé")
        print(compact(assistant, 1500))
    except Exception as e:
        print("❌ Impossible de récupérer l'assistant.")
        show_error(e)
        sys.exit(4)

    section("5. TEST SCHEMA")
    try:
        schemas = await client.assistants.get_schemas(ASSISTANT_ID)
        print("✅ schemas récupérés")
        print(compact(schemas, 3000))
    except Exception as e:
        print("⚠️ Impossible de récupérer les schemas, mais on continue.")
        show_error(e)

    section("6. TEST REMOTEGRAPH AVEC PAYLOAD MINIMAL")
    graph = RemoteGraph(
        ASSISTANT_ID,
        client=client,
    )

    payloads = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": TEST_MESSAGE,
                }
            ]
        },
        {
            "messages": [
                {
                    "type": "human",
                    "content": TEST_MESSAGE,
                }
            ]
        },
    ]

    for i, payload in enumerate(payloads, start=1):
        print(f"\n--- Payload test {i} ---")
        print(compact(payload, 1000))

        try:
            result = await graph.ainvoke(payload)
            print("✅ REMOTEGRAPH INVOKE OK")
            print(compact(result, 4000))
            return
        except Exception as e:
            print("❌ Payload KO")
            show_error(e)

    section("RESULTAT FINAL")
    print("Le certificat et/ou la connexion SDK peuvent être OK, mais aucun payload testé n'est accepté.")
    print("Dans ce cas, regarde surtout la section 5 SCHEMA pour construire le payload exact attendu.")


asyncio.run(main())
PY
