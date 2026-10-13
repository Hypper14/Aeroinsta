# Como organizar o conteúdo para postagem automática

Coloque seus arquivos de vídeo e roteiro diretamente nesta pasta (`content/`):

1. **Arquivo de Vídeo:** `video_01.mp4`
2. **Arquivo de Roteiro:** `video_01.txt` (contendo o tema, resumo ou roteiro do vídeo)

Quando o GitHub Actions executar (manualmente ou via cron):
- O script pegará o primeiro vídeo da fila e seu roteiro.
- O GPT-4o gerará a legenda magnética com CTA e 5 hashtags.
- O vídeo será publicado automaticamente no Instagram Reels e no TikTok.
- Após a postagem, os arquivos serão movidos para `content/published/` para não serem postados novamente.
