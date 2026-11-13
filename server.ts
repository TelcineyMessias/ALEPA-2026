import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import { GoogleGenAI } from '@google/genai';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const GEMINI_MODEL = 'gemini-4-hydrogen';

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('GEMINI_KEY_MISSING');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractYouTubeId(rawUrl: string): string | null {
  const url = rawUrl.trim();
  if (!url) return null;
  const watch = url.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
  if (watch) return watch[1];
  const short = url.match(/youtu\.be\/([a-zA-Z0-9_-]{11})/);
  if (short) return short[1];
  const pathMatch = url.match(/youtube\.com\/(?:shorts|live|embed)\/([a-zA-Z0-9_-]{11})/);
  if (pathMatch) return pathMatch[1];
  return null;
}

function normalizeUrlInput(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
}

function formatGeminiError(err: unknown): { status: number; message: string } {
  const rawMessage = err instanceof Error ? err.message : String(err);
  if (rawMessage === 'GEMINI_KEY_MISSING') {
    return {
      status: 503,
      message:
        'A chave da API Gemini ainda não está disponível neste ambiente. Verifique o painel Settings > Secrets.',
    };
  }
  const lower = rawMessage.toLowerCase();
  if (lower.includes('api_key_invalid') || lower.includes('api key not valid')) {
    return {
      status: 400,
      message: 'A chave da API Gemini configurada está inválida. Verifique no painel Settings > Secrets.',
    };
  }
  if (lower.includes('permission_denied') || lower.includes('403')) {
    return {
      status: 403,
      message: 'Permissão negada ao acessar o Gemini. Verifique a chave configurada em Settings > Secrets.',
    };
  }
  if (lower.includes('resource_exhausted') || lower.includes('429') || lower.includes('quota')) {
    return {
      status: 429,
      message: 'Limite temporário de uso da API Gemini atingido. Aguarde alguns instantes e tente novamente.',
    };
  }
  if (lower.includes('not_found') || lower.includes('404')) {
    return {
      status: 404,
      message: 'Modelo ou recurso não encontrado no Gemini. Tente novamente em instantes.',
    };
  }
  return {
    status: 500,
    message: rawMessage || 'Não foi possível gerar o resumo com o Gemini agora.',
  };
}

function extractGroundingLinks(response: unknown): Array<{ title: string; uri: string }> {
  const res = response as {
    candidates?: Array<{
      groundingMetadata?: {
        groundingChunks?: Array<{
          web?: { uri?: string; title?: string };
        }>;
      };
    }>;
  };
  const chunks = res?.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
  const seen = new Set<string>();
  const links: Array<{ title: string; uri: string }> = [];
  for (const chunk of chunks) {
    const uri = chunk?.web?.uri;
    if (uri && !seen.has(uri)) {
      seen.add(uri);
      links.push({
        uri,
        title: chunk.web?.title || uri,
      });
    }
  }
  return links;
}

const VIDEO_SYSTEM_INSTRUCTION = `Você é um tutor especialista em concursos públicos brasileiros (especialmente ALEPA 2026 · Banca Fundação CETAP).
Sua tarefa é resumir o conteúdo de videoaulas em português do Brasil claro, objetivo e didático para estudo rápido.

Formato recomendado (use quando os tempos e tópicos estiverem disponíveis no vídeo ou na transcrição):
1. Comece com um parágrafo curto de abertura explicando o tema central e eventuais mnemônicos (ex.: "O vídeo explica ... (0:00-0:37).").
2. Liste os principais tópicos em blocos claros no padrão:
   **Nome do Tópico (MM:SS-MM:SS):** Explicação direta, precisa e focada em o que cai em prova.
3. Finalize com uma seção curta **"Pontos de Atenção · Banca CETAP"** com 3 a 5 dicas objetivas de revisão/pegadinhas.

Regras:
- Nunca invente minutagens se elas não puderem ser observadas no vídeo ou no texto enviado; se não houver minutagens exatas disponíveis, organize por tópicos numerados sem inventar horários.
- Seja fiel ao conteúdo jurídico/técnico e destaque artigos constitucionais/legais citados.`;

const PDF_SYSTEM_INSTRUCTION = `Você é um tutor especialista em concursos públicos brasileiros (ALEPA 2026 · Banca Fundação CETAP).
Sua tarefa é resumir apostilas, editais, leis e materiais em PDF em português do Brasil claro e estruturado.

Formato da resposta:
1. **Visão Geral do Documento:** Síntese em 2 a 4 linhas do tema central e objetivo do material.
2. **Resumo Estruturado por Tópicos / Seções:** Organize os principais conceitos, regras, prazos, artigos e definições em tópicos claros com **títulos em negrito**.
3. **Artigos, Prazos e Exceções Importantes:** Liste números, quóruns, competências ou exceções legais relevantes para questões objetivas.
4. **Resumo Relâmpago para Revisão (CETAP):** 3 a 5 bullets curtos com o que mais costuma ser cobrado em prova.`;

async function createApp() {
  const app = express();
  app.use(express.json({ limit: '28mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ ok: true, model: GEMINI_MODEL });
  });

  app.post('/api/gemini/summarize-video', async (req: Request, res: Response) => {
    try {
      const body = (req.body || {}) as {
        url?: string;
        title?: string;
        transcriptOrNotes?: string;
        disciplineName?: string;
        customFocus?: string;
      };

      const rawUrl = typeof body.url === 'string' ? normalizeUrlInput(body.url) : '';
      const title = typeof body.title === 'string' ? body.title.trim() : '';
      const transcriptOrNotes =
        typeof body.transcriptOrNotes === 'string' ? body.transcriptOrNotes.trim() : '';
      const disciplineName = typeof body.disciplineName === 'string' ? body.disciplineName.trim() : '';
      const customFocus = typeof body.customFocus === 'string' ? body.customFocus.trim() : '';

      if (!rawUrl && !transcriptOrNotes) {
        res.status(400).json({
          error: 'Informe o link do vídeo (YouTube, Instagram, TikTok) ou cole anotações/transcrição para resumir.',
        });
        return;
      }

      const ai = getGeminiClient();
      const ytId = rawUrl ? extractYouTubeId(rawUrl) : null;
      const canonicalYtUrl = ytId ? `https://www.youtube.com/watch?v=${ytId}` : null;

      const userPromptParts = [
        title ? `Título informado: ${title}` : '',
        disciplineName ? `Disciplina de estudo: ${disciplineName}` : '',
        rawUrl ? `Link do vídeo: ${canonicalYtUrl || rawUrl}` : '',
        customFocus ? `Foco pedido pelo estudante: ${customFocus}` : '',
        transcriptOrNotes
          ? `Anotações / transcrição fornecida pelo estudante:\n${transcriptOrNotes}`
          : 'Gere um resumo didático completo por tópicos (com intervalos de tempo MM:SS-MM:SS sempre que identificáveis no vídeo).',
      ].filter(Boolean);
      const userPrompt = userPromptParts.join('\n\n');

      // 1. Direct multimodal YouTube summarization when a YouTube video ID is present
      if (canonicalYtUrl) {
        try {
          const response = await ai.models.generateContent({
            model: GEMINI_MODEL,
            contents: [
              {
                role: 'user',
                parts: [
                  {
                    fileData: {
                      fileUri: canonicalYtUrl,
                      mimeType: 'video/mp4',
                    },
                  },
                  { text: userPrompt },
                ],
              },
            ],
            config: {
              systemInstruction: VIDEO_SYSTEM_INSTRUCTION,
            },
          });

          const text = response.text?.trim();
          if (text) {
            res.json({
              summary: text,
              model: GEMINI_MODEL,
              sourceMode: 'youtube_video',
              sources: [{ title: title || 'Vídeo no YouTube', uri: canonicalYtUrl }],
            });
            return;
          }
        } catch (ytErr) {
          console.warn('Fallback após análise direta do YouTube:', ytErr instanceof Error ? ytErr.message : ytErr);
        }
      }

      // 2. Fallback or non-YouTube URL: urlContext + googleSearch, or direct notes summarization
      if (rawUrl) {
        try {
          const response = await ai.models.generateContent({
            model: GEMINI_MODEL,
            contents: `${userPrompt}\n\nSe não for possível ler diretamente o áudio/vídeo deste link externo, deixe claro o que pôde ser obtido pela página/pesquisa ou pelas anotações enviadas, e ajude o estudante com uma explicação completa do tema identificado.`,
            config: {
              systemInstruction: VIDEO_SYSTEM_INSTRUCTION,
              tools: [{ urlContext: {} }, { googleSearch: {} }],
            },
          });
          const text = response.text?.trim();
          if (text) {
            const links = extractGroundingLinks(response);
            res.json({
              summary: text,
              model: GEMINI_MODEL,
              sourceMode: 'url_context',
              sources: links.length > 0 ? links : [{ title: title || 'Link do vídeo', uri: rawUrl }],
            });
            return;
          }
        } catch (urlErr) {
          if (!transcriptOrNotes) {
            throw urlErr;
          }
        }
      }

      // 3. Direct summarization from pasted transcript / notes
      const fallbackResponse = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: userPrompt,
        config: {
          systemInstruction: VIDEO_SYSTEM_INSTRUCTION,
        },
      });

      res.json({
        summary: fallbackResponse.text?.trim() || 'Não foi possível gerar texto para este vídeo.',
        model: GEMINI_MODEL,
        sourceMode: 'notes_only',
        sources: rawUrl ? [{ title: title || 'Link informado', uri: rawUrl }] : [],
      });
    } catch (err) {
      const mapped = formatGeminiError(err);
      res.status(mapped.status).json({ error: mapped.message });
    }
  });

  app.post('/api/gemini/summarize-pdf', async (req: Request, res: Response) => {
    try {
      const body = (req.body || {}) as {
        fileBase64?: string;
        fileName?: string;
        pdfUrl?: string;
        title?: string;
        disciplineName?: string;
        customFocus?: string;
      };

      const fileName = typeof body.fileName === 'string' ? body.fileName.trim() : 'documento.pdf';
      const title = typeof body.title === 'string' ? body.title.trim() : '';
      const disciplineName = typeof body.disciplineName === 'string' ? body.disciplineName.trim() : '';
      const customFocus = typeof body.customFocus === 'string' ? body.customFocus.trim() : '';
      const rawPdfUrl = typeof body.pdfUrl === 'string' ? normalizeUrlInput(body.pdfUrl) : '';
      let base64Data = typeof body.fileBase64 === 'string' ? body.fileBase64.trim() : '';

      if (base64Data.includes(',')) {
        base64Data = base64Data.split(',')[1] || '';
      }

      if (!base64Data && !rawPdfUrl) {
        res.status(400).json({
          error: 'Envie um arquivo PDF ou informe o link do PDF/material para resumir.',
        });
        return;
      }

      const ai = getGeminiClient();

      const promptLines = [
        title ? `Título do material: ${title}` : '',
        fileName ? `Arquivo: ${fileName}` : '',
        disciplineName ? `Disciplina: ${disciplineName}` : '',
        customFocus ? `Foco pedido pelo estudante: ${customFocus}` : '',
        'Faça um resumo completo, fiel e bem organizado deste PDF para revisão de concurso.',
      ].filter(Boolean);
      const promptText = promptLines.join('\n');

      // Try fetching remote PDF bytes if pdfUrl was supplied without inline base64
      let fetchedUrlMode: 'pdf_inline' | 'url_context' = 'pdf_inline';
      if (!base64Data && rawPdfUrl) {
        let fetchTarget = rawPdfUrl;
        const driveMatch = rawPdfUrl.match(/drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)/i);
        if (driveMatch) {
          fetchTarget = `https://drive.google.com/uc?export=download&id=${driveMatch[1]}`;
        }

        try {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 15000);
          const remoteRes = await fetch(fetchTarget, { signal: controller.signal });
          clearTimeout(timer);

          if (remoteRes.ok) {
            const ctype = (remoteRes.headers.get('content-type') || '').toLowerCase();
            const arrBuf = await remoteRes.arrayBuffer();
            const buf = Buffer.from(arrBuf);
            const looksLikePdf =
              ctype.includes('application/pdf') || buf.subarray(0, 4).toString('utf8') === '%PDF';

            if (looksLikePdf && buf.byteLength <= 18 * 1024 * 1024) {
              base64Data = buf.toString('base64');
              fetchedUrlMode = 'pdf_inline';
            } else {
              fetchedUrlMode = 'url_context';
            }
          } else {
            fetchedUrlMode = 'url_context';
          }
        } catch {
          fetchedUrlMode = 'url_context';
        }
      }

      if (base64Data) {
        const approxBytes = Math.floor((base64Data.length * 3) / 4);
        if (approxBytes > 20 * 1024 * 1024) {
          res.status(413).json({
            error: 'O PDF excede o limite de 20 MB para envio direto. Divida o arquivo em partes menores.',
          });
          return;
        }

        const response = await ai.models.generateContent({
          model: GEMINI_MODEL,
          contents: [
            {
              role: 'user',
              parts: [
                {
                  inlineData: {
                    mimeType: 'application/pdf',
                    data: base64Data,
                  },
                },
                { text: promptText },
              ],
            },
          ],
          config: {
            systemInstruction: PDF_SYSTEM_INSTRUCTION,
          },
        });

        res.json({
          summary: response.text?.trim() || 'Não foi possível extrair texto deste PDF.',
          model: GEMINI_MODEL,
          sourceMode: fetchedUrlMode,
          sources: rawPdfUrl ? [{ title: title || fileName, uri: rawPdfUrl }] : [],
        });
        return;
      }

      // Fallback to urlContext for web pages / legislation links / Drive preview links
      const urlResponse = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: `${promptText}\n\nLink do material: ${rawPdfUrl}`,
        config: {
          systemInstruction: PDF_SYSTEM_INSTRUCTION,
          tools: [{ urlContext: {} }],
        },
      });

      const links = extractGroundingLinks(urlResponse);
      res.json({
        summary: urlResponse.text?.trim() || 'Não foi possível ler o link informado.',
        model: GEMINI_MODEL,
        sourceMode: 'url_context',
        sources: links.length > 0 ? links : [{ title: title || rawPdfUrl, uri: rawPdfUrl }],
      });
    } catch (err) {
      const mapped = formatGeminiError(err);
      res.status(mapped.status).json({ error: mapped.message });
    }
  });

  app.post('/api/gemini/chat', async (req: Request, res: Response) => {
    try {
      const body = (req.body || {}) as {
        messages?: Array<{ role?: string; text?: string }>;
        contextTitle?: string;
        contextSummary?: string;
        campus?: string;
      };

      const rawMessages = Array.isArray(body.messages) ? body.messages : [];
      const contents = rawMessages
        .filter((m) => typeof m?.text === 'string' && m.text.trim().length > 0)
        .slice(-16)
        .map((m) => ({
          role: m.role === 'model' ? ('model' as const) : ('user' as const),
          parts: [{ text: (m.text || '').trim() }],
        }));

      if (contents.length === 0) {
        res.status(400).json({ error: 'Digite uma pergunta para conversar com o Gemini.' });
        return;
      }

      const ai = getGeminiClient();
      const campusLabel =
        body.campus === 'medio'
          ? 'Nível Médio (Cargo 04 – Assistência Administrativa · PL.AL.060)'
          : 'Nível Superior (Cargo 03 – Assessoramento Legislativo · PL.AL.102)';

      const systemParts = [
        `Você é o assistente de estudos Gemini para o Concurso ALEPA 2026 (Banca Fundação CETAP), foco em ${campusLabel}.`,
        'Responda sempre em português do Brasil de forma clara, estruturada e objetiva.',
        body.contextTitle ? `Material em foco: ${body.contextTitle}` : '',
        body.contextSummary
          ? `Use também como referência o resumo já gerado abaixo:\n---\n${body.contextSummary.slice(0, 8000)}\n---`
          : '',
      ].filter(Boolean);

      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents,
        config: {
          systemInstruction: systemParts.join('\n\n'),
        },
      });

      res.json({
        reply: response.text?.trim() || 'Não consegui gerar uma resposta agora.',
        model: GEMINI_MODEL,
      });
    } catch (err) {
      const mapped = formatGeminiError(err);
      res.status(mapped.status).json({ error: mapped.message });
    }
  });

  if (process.env.NODE_ENV === 'production') {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      const indexPath = path.join(distDir, 'index.html');
      if (fs.existsSync(indexPath)) {
        res.sendFile(indexPath);
      } else {
        res.status(404).send('Build not found.');
      }
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

createApp().catch((err) => {
  console.error('Erro fatal ao iniciar servidor:', err);
  process.exit(1);
});
