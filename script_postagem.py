#!/usr/bin/env python3
"""
Agente Autônomo de Postagem Multiplataforma (Instagram Reels & TikTok)
=====================================================================
Desenvolvido para execução automatizada via GitHub Actions.

Fluxo:
1. Localiza o próximo vídeo (.mp4) e roteiro (.txt) na pasta de conteúdo.
2. Gera legenda persuasiva, CTA e 5 hashtags otimizadas via OpenAI (GPT-4o).
3. Publica o vídeo no TikTok via TikTok Content Posting API (Direct Post).
4. Publica o vídeo no Instagram Reels via Meta Graph API (Resumable Upload)
   para a conta @victorn_13 (com auto-resolução do ID numérico da conta).
5. Move os arquivos processados para a pasta de arquivamento (published/).
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis de ambiente (.env para testes locais)
load_dotenv()

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AutoPoster")


class Config:
    """Configurações e validação das variáveis de ambiente."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    INSTAGRAM_ACCESS_TOKEN: str = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
    INSTAGRAM_ACCOUNT_ID: str = os.getenv("INSTAGRAM_ACCOUNT_ID", "")
    INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "victorn_13")
    TIKTOK_ACCESS_TOKEN: str = os.getenv("TIKTOK_ACCESS_TOKEN", "")
    
    # Configurações opcionais / personalizáveis
    CONTENT_DIR: Path = Path(os.getenv("CONTENT_DIR", "content"))
    ARCHIVE_DIR: Path = Path(os.getenv("ARCHIVE_DIR", "content/published"))
    TIKTOK_PRIVACY_LEVEL: str = os.getenv("TIKTOK_PRIVACY_LEVEL", "PUBLIC_TO_EVERYONE")
    META_GRAPH_VERSION: str = os.getenv("META_GRAPH_VERSION", "v21.0")

    @classmethod
    def validate(cls) -> None:
        """Valida a presença de todos os segredos obrigatórios."""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.INSTAGRAM_ACCESS_TOKEN:
            missing.append("INSTAGRAM_ACCESS_TOKEN")
        if not cls.TIKTOK_ACCESS_TOKEN:
            missing.append("TIKTOK_ACCESS_TOKEN")

        if missing:
            logger.error(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
            logger.error("Configure essas variáveis nos GitHub Secrets do repositório.")
            sys.exit(1)


# ==============================================================================
# 1. LOCALIZAÇÃO E LEITURA DOS ARQUIVOS
# ==============================================================================

def find_content_pair(content_dir: Path) -> Tuple[Path, Path]:
    """
    Localiza o primeiro par de arquivos (.mp4 e .txt) na pasta de conteúdo.
    Prioriza o .txt com o mesmo nome base do vídeo.
    """
    if not content_dir.exists():
        content_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"A pasta '{content_dir}' foi criada, mas não contém arquivos. "
            "Adicione um arquivo .mp4 e um .txt para continuar."
        )

    # Lista apenas arquivos de vídeo na pasta raiz de conteúdo
    video_files = sorted([
        f for f in content_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    ])

    if not video_files:
        raise FileNotFoundError(f"Nenhum arquivo de vídeo (.mp4) encontrado em '{content_dir}'.")

    video_path = video_files[0]
    
    # Busca .txt com o mesmo nome do vídeo ou o primeiro .txt disponível
    matching_txt = video_path.with_suffix(".txt")
    if matching_txt.exists():
        text_path = matching_txt
    else:
        text_files = sorted([
            f for f in content_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        ])
        if not text_files:
            raise FileNotFoundError(f"Nenhum arquivo de roteiro (.txt) encontrado em '{content_dir}'.")
        text_path = text_files[0]

    logger.info(f"Vídeo selecionado: {video_path.name}")
    logger.info(f"Roteiro selecionado: {text_path.name}")
    return video_path, text_path


def read_script_content(file_path: Path) -> str:
    """Lê e retorna o conteúdo do arquivo de roteiro/tema."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"O arquivo {file_path.name} está vazio.")
        return content
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()


# ==============================================================================
# 2. GERAÇÃO DE LEGENDA COM OPENAI (GPT-4o)
# ==============================================================================

def generate_caption_with_gpt4o(script_text: str, api_key: str) -> str:
    """
    Envia o roteiro para o GPT-4o e gera uma legenda magnética
    com gancho, contexto, CTA e exatamente 5 hashtags ideais para Reels e TikTok.
    """
    logger.info("Solicitando geração de legenda ao GPT-4o...")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "Você é um copywriter e estrategista de social media sênior, especialista "
        "em conteúdos virais de alta retenção e conversão para Instagram Reels e TikTok.\n\n"
        "Sua missão é criar uma legenda magnética a partir do roteiro ou tema fornecido pelo usuário.\n\n"
        "Diretrizes obrigatórias:\n"
        "1. Gancho Inicial (Hook): A primeira linha deve ser irresistível, despertando curiosidade ou tocando em uma dor forte.\n"
        "2. Corpo da Legenda: Entregue contexto de forma concisa, dinâmica e escaneável (use quebras de linha e emojis com moderação).\n"
        "3. Chamada para Ação (CTA): Uma chamada clara incentivando o usuário a comentar, salvar ou compartilhar.\n"
        "4. Hashtags: Finalize com EXATAMENTE 5 hashtags altamente estratégicas e relevantes para o nicho (misturando hashtags amplas e específicas).\n"
        "5. Formato de Saída: Retorne APENAS o texto final da legenda pronto para publicação. Não inclua introduções, explicações ou aspas."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Roteiro/Tema do Vídeo:\n\n{script_text}"}
            ],
            temperature=0.7,
            max_tokens=600
        )
        caption = response.choices[0].message.content.strip()
        logger.info("Legenda gerada com sucesso pela OpenAI.")
        logger.info(f"\n--- LEGENDA GERADA ---\n{caption}\n----------------------")
        return caption
    except Exception as e:
        logger.error(f"Erro ao chamar a API da OpenAI: {e}")
        raise


# ==============================================================================
# 3. PUBLICAÇÃO NO TIKTOK (Content Posting API v2)
# ==============================================================================

def publish_to_tiktok(video_path: Path, caption: str, access_token: str, privacy_level: str = "PUBLIC_TO_EVERYONE") -> str:
    """
    Publica o vídeo no TikTok utilizando a Content Posting API oficial (Direct Post).
    Fluxo: Init -> Upload Binary -> Status Check.
    """
    logger.info("Iniciando publicação no TikTok via Content Posting API...")
    file_size = video_path.stat().st_size
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8"
    }

    # Limite oficial de título/legenda do TikTok: máximo de 2200 caracteres
    title = caption[:2200]

    # Passo 1: Inicializar publicação
    init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    init_payload = {
        "post_info": {
            "title": title,
            "privacy_level": privacy_level,
            "disable_duet": False,
            "disable_stitch": False,
            "disable_comment": False,
            "video_cover_timestamp_ms": 1000
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": file_size,
            "total_chunk_count": 1
        }
    }

    resp = requests.post(init_url, headers=headers, json=init_payload)
    if resp.status_code != 200:
        logger.error(f"Erro ao inicializar post no TikTok: {resp.status_code} - {resp.text}")
        raise RuntimeError(f"TikTok Init Error: {resp.text}")

    init_data = resp.json().get("data", {})
    publish_id = init_data.get("publish_id")
    upload_url = init_data.get("upload_url")

    if not upload_url or not publish_id:
        logger.error(f"Resposta inválida do TikTok: {resp.json()}")
        raise RuntimeError(f"Dados de upload não encontrados: {resp.text}")

    logger.info(f"Sessão de upload criada no TikTok. Publish ID: {publish_id}")

    # Passo 2: Enviar arquivo binário
    logger.info(f"Enviando arquivo ({file_size / (1024*1024):.2f} MB) para o TikTok...")
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()

    upload_headers = {
        "Content-Type": "video/mp4",
        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        "Content-Length": str(file_size)
    }

    upload_resp = requests.put(upload_url, headers=upload_headers, data=video_bytes)
    if upload_resp.status_code not in [200, 201]:
        logger.error(f"Erro no upload para o TikTok: {upload_resp.status_code} - {upload_resp.text}")
        raise RuntimeError(f"TikTok Upload Failed: {upload_resp.text}")

    logger.info("Upload de vídeo no TikTok concluído. Verificando status de processamento...")

    # Passo 3: Consultar status de processamento
    status_url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    status_payload = {"publish_id": publish_id}

    for attempt in range(12):
        time.sleep(10)
        status_resp = requests.post(status_url, headers=headers, json=status_payload)
        if status_resp.status_code == 200:
            status_data = status_resp.json().get("data", {})
            status = status_data.get("status")
            logger.info(f"Status TikTok (tentativa {attempt + 1}): {status}")

            if status == "PUBLISH_COMPLETE":
                logger.info(f"Vídeo publicado com sucesso no TikTok! Publish ID: {publish_id}")
                return publish_id
            elif status == "FAILED":
                fail_reason = status_data.get("fail_reason", "Desconhecido")
                raise RuntimeError(f"Falha na publicação no TikTok: {fail_reason}")
            elif status in ["PROCESSING_UPLOAD", "PROCESSING_DOWNLOAD"]:
                continue
        else:
            logger.warning(f"Consulta de status retornou código {status_resp.status_code}: {status_resp.text}")

    logger.info(f"Upload concluído no TikTok e em processamento assíncrono. Publish ID: {publish_id}")
    return publish_id


# ==============================================================================
# 4. PUBLICAÇÃO NO INSTAGRAM REELS (Meta Graph API)
# ==============================================================================

def get_instagram_business_account_id(
    access_token: str,
    target_username: str = "victorn_13",
    api_version: str = "v21.0"
) -> str:
    """
    Busca automaticamente o ID numérico da conta do Instagram conectada
    à Página do Facebook associada ao token de acesso.
    """
    clean_target = target_username.lower().lstrip("@").strip()
    logger.info(f"Buscando ID numérico da conta Instagram para o perfil '@{clean_target}'...")

    url = f"https://graph.facebook.com/{api_version}/me/accounts"
    params = {
        "fields": "id,name,instagram_business_account{id,username}",
        "access_token": access_token
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        logger.error(f"Erro ao consultar Páginas/Contas do Meta Graph: {resp.text}")
        raise RuntimeError(f"Meta Graph Accounts Error: {resp.text}")

    pages = resp.json().get("data", [])
    found_accounts = []

    for page in pages:
        ig_account = page.get("instagram_business_account")
        if ig_account:
            ig_id = ig_account.get("id")
            ig_user = ig_account.get("username", "")
            found_accounts.append((ig_user, ig_id))
            
            if clean_target and ig_user.lower() == clean_target:
                logger.info(f"Conta encontrada: @{ig_user} -> ID: {ig_id}")
                return str(ig_id)

    # Se não especificou ou não encontrou exato mas tem contas conectadas
    if found_accounts:
        first_user, first_id = found_accounts[0]
        logger.info(f"Utilizando conta Instagram encontrada: @{first_user} (ID: {first_id})")
        return str(first_id)

    raise ValueError(
        f"Nenhuma conta de Instagram Business/Creator associada foi encontrada para '@{clean_target}'. "
        "Certifique-se de que a conta @victorn_13 é Profissional e está vinculada a uma Página do Facebook."
    )


def publish_to_instagram_reels(
    video_path: Path,
    caption: str,
    access_token: str,
    instagram_account_id: str = "",
    target_username: str = "victorn_13",
    api_version: str = "v21.0"
) -> str:
    """
    Publica o vídeo no Instagram Reels usando a Meta Graph API oficial.
    Utiliza o método de 'Resumable Upload' para envio direto do arquivo local.
    Fluxo: Resolve Account ID -> Init Container -> Upload Binary -> Wait -> Publish.
    """
    # 1. Resolve o ID da conta se não for um ID numérico direto
    if not instagram_account_id or not instagram_account_id.isdigit():
        target = instagram_account_id if instagram_account_id else target_username
        instagram_account_id = get_instagram_business_account_id(
            access_token=access_token,
            target_username=target,
            api_version=api_version
        )

    logger.info(f"Iniciando publicação no Instagram Reels (Conta ID: {instagram_account_id})...")
    file_size = video_path.stat().st_size
    base_url = f"https://graph.facebook.com/{api_version}"

    # Passo 1: Inicializar container de mídia (Resumable Upload)
    init_url = f"{base_url}/{instagram_account_id}/media"
    init_params = {
        "media_type": "REELS",
        "upload_type": "resumable",
        "caption": caption,
        "access_token": access_token
    }

    init_resp = requests.post(init_url, data=init_params)
    if init_resp.status_code != 200:
        logger.error(f"Erro ao criar container no Instagram: {init_resp.status_code} - {init_resp.text}")
        raise RuntimeError(f"Instagram Container Creation Error: {init_resp.text}")

    init_data = init_resp.json()
    container_id = init_data.get("id")
    upload_uri = init_data.get("uri")

    if not container_id or not upload_uri:
        logger.error(f"Dados incompletos retornados pela Meta: {init_data}")
        raise RuntimeError("Instagram ID ou URI de upload ausente.")

    logger.info(f"Container criado no Instagram. Container ID: {container_id}")

    # Passo 2: Enviar arquivo binário para a URI de upload
    logger.info(f"Enviando arquivo ({file_size / (1024*1024):.2f} MB) para os servidores da Meta...")
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()

    upload_headers = {
        "Authorization": f"OAuth {access_token}",
        "file_offset": "0",
        "Content-Type": "application/octet-stream"
    }

    upload_resp = requests.post(upload_uri, headers=upload_headers, data=video_bytes)
    if upload_resp.status_code not in [200, 201]:
        logger.error(f"Erro ao enviar binário para o Instagram: {upload_resp.status_code} - {upload_resp.text}")
        raise RuntimeError(f"Instagram Video Upload Failed: {upload_resp.text}")

    logger.info("Upload de vídeo no Instagram concluído. Aguardando processamento do container...")

    # Passo 3: Aguardar o processamento do container pela Meta
    status_url = f"{base_url}/{container_id}"
    status_params = {
        "fields": "status_code,status",
        "access_token": access_token
    }

    max_retries = 24  # até 4 minutos
    container_ready = False

    for attempt in range(max_retries):
        time.sleep(10)
        status_resp = requests.get(status_url, params=status_params)
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            status_code = status_data.get("status_code")
            logger.info(f"Status do container Instagram (tentativa {attempt + 1}): {status_code}")

            if status_code == "FINISHED":
                container_ready = True
                break
            elif status_code in ["ERROR", "EXPIRED"]:
                error_msg = status_data.get("status", "Erro no processamento do vídeo.")
                raise RuntimeError(f"Falha no processamento do vídeo pelo Instagram: {error_msg}")
            elif status_code == "IN_PROGRESS":
                continue
        else:
            logger.warning(f"Erro ao checar status do container: {status_resp.text}")

    if not container_ready:
        raise TimeoutError("Tempo esgotado aguardando o processamento do vídeo no Instagram.")

    # Passo 4: Publicar o container
    logger.info("Container pronto. Executando publicação final...")
    publish_url = f"{base_url}/{instagram_account_id}/media_publish"
    publish_params = {
        "creation_id": container_id,
        "access_token": access_token
    }

    publish_resp = requests.post(publish_url, data=publish_params)
    if publish_resp.status_code != 200:
        logger.error(f"Erro ao publicar mídia no Instagram: {publish_resp.status_code} - {publish_resp.text}")
        raise RuntimeError(f"Instagram Publish Error: {publish_resp.text}")

    published_id = publish_resp.json().get("id")
    logger.info(f"Reels publicado com sucesso no Instagram (@victorn_13)! Media ID: {published_id}")
    return published_id


# ==============================================================================
# 5. ARQUIVAMENTO DOS ARQUIVOS PROCESSADOS
# ==============================================================================

def archive_processed_files(video_path: Path, text_path: Path, archive_dir: Path) -> None:
    """Move os arquivos processados para a pasta de histórico/publicados."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest_video = archive_dir / f"{timestamp}_{video_path.name}"
    dest_text = archive_dir / f"{timestamp}_{text_path.name}"

    shutil.move(str(video_path), str(dest_video))
    shutil.move(str(text_path), str(dest_text))

    logger.info(f"Arquivos arquivados com sucesso em '{archive_dir}':")
    logger.info(f" - {dest_video.name}")
    logger.info(f" - {dest_text.name}")


# ==============================================================================
# FLUXO PRINCIPAL
# ==============================================================================

def main():
    logger.info("=== AGENTE AUTÔNOMO DE POSTAGEM INICIADO ===")
    
    # 1. Valida variáveis de ambiente
    Config.validate()

    # 2. Localiza arquivos
    try:
        video_path, text_path = find_content_pair(Config.CONTENT_DIR)
    except FileNotFoundError as e:
        logger.warning(f"Nenhum conteúdo para postar: {e}")
        logger.info("Finalizando execução sem erros.")
        sys.exit(0)

    # 3. Lê o roteiro
    script_content = read_script_content(text_path)
    logger.info(f"Conteúdo do roteiro ({len(script_content)} caracteres) lido com sucesso.")

    # 4. Gera a legenda com OpenAI GPT-4o
    caption = generate_caption_with_gpt4o(script_content, Config.OPENAI_API_KEY)

    # 5. Publicação no TikTok
    tiktok_success = False
    try:
        publish_to_tiktok(
            video_path=video_path,
            caption=caption,
            access_token=Config.TIKTOK_ACCESS_TOKEN,
            privacy_level=Config.TIKTOK_PRIVACY_LEVEL
        )
        tiktok_success = True
    except Exception as e:
        logger.error(f"[TIKTOK] Falha na publicação: {e}")

    # 6. Publicação no Instagram Reels (@victorn_13)
    instagram_success = False
    try:
        publish_to_instagram_reels(
            video_path=video_path,
            caption=caption,
            access_token=Config.INSTAGRAM_ACCESS_TOKEN,
            instagram_account_id=Config.INSTAGRAM_ACCOUNT_ID,
            target_username=Config.INSTAGRAM_USERNAME,
            api_version=Config.META_GRAPH_VERSION
        )
        instagram_success = True
    except Exception as e:
        logger.error(f"[INSTAGRAM] Falha na publicação: {e}")

    # 7. Resumo e Arquivamento
    if tiktok_success or instagram_success:
        logger.info("Pelo menos uma publicação foi bem-sucedida. Arquivando arquivos...")
        archive_processed_files(video_path, text_path, Config.ARCHIVE_DIR)
        logger.info("=== PROCESSO CONCLUÍDO COM SUCESSO ===")
    else:
        logger.error("Nenhuma publicação teve êxito. Os arquivos foram mantidos para nova tentativa.")
        sys.exit(1)


if __name__ == "__main__":
    main()
