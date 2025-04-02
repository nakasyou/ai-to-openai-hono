/**
 * @example
 * ```ts
 * import { Hono } from 'hono'
 * import { createOpenAIHono } from '@ns/ai-to-openai-hono'
 *
 * import { anthropic } from  'npm:@ai-sdk/anthropic' // or your favorite provider
 *
 * const app = new Hono()
 *
 * app.route('/my-ai-endpoint', createOpenAIHono({
 *   languageModels: {
 *     'claude-3.7-sonnet': anthropic('claude-3-7-sonnet-20250219') // or your favorite model,
 *     // ...
 *   },
 *   verifyAPIKey(key) {
 *     return key === 'this-is-super-secret-key'
 *   }
 * }))
 *
 * export default app
 * ```
 * @module
 */

import { Hono } from 'hono'
import type {
  LanguageModelV1,
  LanguageModelV1FinishReason,
} from '@ai-sdk/provider'
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
} from 'openai/resources/chat/completions'
import {
  type CoreMessage,
  type FilePart,
  generateText,
  type ImagePart,
  streamText as streamTextAI,
  type TextPart,
} from 'ai'
import { streamText as streamTextHono } from 'hono/streaming'
import { HTTPException } from 'hono/http-exception'

const STOP_REASON: Record<
  LanguageModelV1FinishReason,
  'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call' | null
> = {
  length: 'length',
  stop: 'stop',
  'content-filter': 'content_filter',
  'tool-calls': 'tool_calls',
  unknown: 'stop',
  error: 'stop',
  other: 'stop',
}

/**
 * Options to init
 */
export interface Init {
  languageModels:
    | Record<string, LanguageModelV1>
    | ((
      modelId: string,
    ) => Promise<LanguageModelV1 | null> | LanguageModelV1 | null)

  /**
   * @default `() => true`
   */
  verifyAPIKey?: (key: string) => boolean | Promise<boolean>
}

type SDKInit =
  & Parameters<typeof generateText>[0]
  & Parameters<typeof streamTextAI>[0]

/**
 * Create Hono App from AI SDK
 * @param init Options to init
 * @returns Hono app
 */
export const createOpenAICompat = (init: Init): Hono => {
  const app = new Hono()

  if (init.verifyAPIKey) {
    app.use(async (c, next) => {
      const authHeader = c.req.header('Authorization')
      if (!authHeader) {
        throw new HTTPException(403, {
          message: 'Missing Authorization header',
        })
      }
      const key = authHeader.split(' ')[1]
      const isValid = await init.verifyAPIKey?.(key)
      if (!isValid) {
        throw new HTTPException(403, {
          message: 'Invalid API key',
        })
      }
      await next()
    })
  }

  app.post('/v1/chat/completions', async (c) => {
    const body = await c.req.json<ChatCompletionCreateParams>()

    const model = typeof init.languageModels === 'function'
      ? await init.languageModels(body.model)
      : init.languageModels[body.model]
    if (!model) {
      throw new HTTPException(400, {
        message: 'Invalid model',
      })
    }
    const aiSDKInit: SDKInit = {
      model,
      messages: body.messages.map((message): CoreMessage => {
        if (message.role == 'assistant') {
          return {
            role: 'assistant',
            content: message.content
              ? typeof message.content === 'string'
                ? message.content
                : message.content.map((c) =>
                  c.type === 'text' ? c.text : c.refusal
                ).join('')
              : '',
          }
        } else if (message.role == 'user') {
          return {
            role: 'user',
            content: typeof message.content === 'string'
              ? message.content
              : message.content.map((c): TextPart | ImagePart | FilePart => {
                if (c.type === 'text') {
                  return {
                    type: 'text',
                    text: c.text,
                  }
                }
                if (c.type === 'image_url') {
                  return {
                    type: 'image',
                    image: new URL(c.image_url.url),
                  }
                }
                if (c.type === 'file') {
                  return {
                    type: 'file',
                    data: c.file.file_data ?? '',
                    mimeType: '',
                  }
                }
                return {
                  type: 'file',
                  data: c.input_audio.data,
                  mimeType: c.input_audio.format === 'mp3'
                    ? 'audio/mpeg'
                    : 'audio/wav',
                }
              }),
          }
        } else if (message.role === 'developer') {
          return {
            role: 'system',
            content: typeof message.content === 'string'
              ? message.content
              : message.content.map((c) => c.text).join(''),
          }
        } else if (message.role === 'function') {
          return {
            role: 'system',
            content: message.content ?? '',
          }
        } else if (message.role === 'system') {
          return {
            role: 'system',
            content: typeof message.content === 'string'
              ? message.content
              : message.content.map((c) => c.text).join(''),
          }
        } else if (message.role === 'tool') {
          return {
            role: 'tool',
            content: typeof message.content === 'string'
              ? []
              : message.content.map((c) => ({
                type: 'tool-result',
                toolCallId: message.tool_call_id,
                toolName: message.tool_call_id,
                result: c.text,
              })),
          }
        }
        throw new Error(`Unreachable`)
      }),
      temperature: body.temperature ?? undefined,
      topP: body.top_p ?? undefined,
      frequencyPenalty: body.frequency_penalty ?? undefined,
      presencePenalty: body.presence_penalty ?? undefined,
      maxTokens: body.max_tokens ?? undefined,
      stopSequences: body.stop
        ? typeof body.stop === 'string' ? [body.stop] : body.stop
        : undefined,
      tools: body.tools
        ? Object.fromEntries(
          body.tools?.map((tool) => [
            tool.function.name,
            {
              type: 'function',
              description: tool.function.description,
              parameters: tool.function.parameters,
            },
          ]),
        )
        : undefined,
      toolChoice: body.tool_choice
        ? (typeof body.tool_choice === 'string' ? body.tool_choice : {
          type: 'tool',
          toolName: body.tool_choice.function.name,
        })
        : undefined,
      seed: body.seed ?? undefined,
    }
    if (body.stream) {
      const aiStream = streamTextAI(aiSDKInit)

      return streamTextHono(c, async (stream) => {
        const streamChunk = async (data: ChatCompletionChunk) => {
          await stream.write(`data: ${JSON.stringify(data)}\n\n`)
        }

        const reader = aiStream.fullStream.getReader()
        while (true) {
          const { value, done } = await reader.read()
          if (value) {
            if (value.type === 'error') {
              await streamChunk({
                id: crypto.randomUUID(),
                object: 'chat.completion.chunk',
                created: Date.now() / 1000,
                model: aiSDKInit.model.modelId,
                choices: [{
                  index: 0,
                  delta: {},
                  finish_reason: 'stop',
                }],
              })
              break
            } else if (value.type === 'finish') {
              await streamChunk({
                id: crypto.randomUUID(),
                object: 'chat.completion.chunk',
                created: Date.now() / 1000,
                model: aiSDKInit.model.modelId,
                choices: [{
                  index: 0,
                  delta: {},
                  finish_reason: STOP_REASON[value.finishReason],
                }],
                usage: value.usage
                  ? {
                    completion_tokens: value.usage.completionTokens,
                    prompt_tokens: value.usage.promptTokens,
                    total_tokens: value.usage.totalTokens,
                  }
                  : undefined,
              })
              break
            } else if (value.type === 'text-delta') {
              await streamChunk({
                id: crypto.randomUUID(),
                object: 'chat.completion.chunk',
                created: Date.now() / 1000,
                model: aiSDKInit.model.modelId,
                choices: [{
                  index: 0,
                  delta: {
                    role: 'assistant',
                    content: value.textDelta,
                  },
                  finish_reason: null,
                }],
              })
            } else if (value.type === 'tool-call') {
              await streamChunk({
                id: crypto.randomUUID(),
                object: 'chat.completion.chunk',
                created: Date.now() / 1000,
                model: aiSDKInit.model.modelId,
                choices: [{
                  index: 0,
                  delta: {
                    role: 'assistant',
                    tool_calls: [{
                      id: value.toolCallId,
                      index: 0,
                      function: {
                        name: value.toolName,
                        arguments: value.args,
                      },
                    }],
                  },
                  finish_reason: 'stop',
                }],
              })
              break
            }
          }
          if (done) {
            break
          }
        }

        await stream.write('data: [DONE]\n\n')
        await stream.close()
      })
    }

    const generated = await generateText(aiSDKInit)
    const resultJSON: ChatCompletion = {
      id: generated.response.id,
      object: 'chat.completion',
      created: generated.response.timestamp.getTime() / 1000,
      model: generated.response.modelId,
      choices: [{
        index: 0,
        //@ts-expect-error idk
        finish_reason: STOP_REASON[generated.finishReason],
        logprobs: null,
        message: {
          role: 'assistant',
          content: generated.text,
          refusal: '',
          tool_calls: generated.toolCalls.map((call) => ({
            id: call.toolCallId,
            type: 'function',
            function: {
              name: call.toolName,
              arguments: call.args,
            },
          })),
        },
      }],
      usage: generated.usage
        ? {
          completion_tokens: generated.usage.completionTokens,
          prompt_tokens: generated.usage.promptTokens,
          total_tokens: generated.usage.totalTokens,
        }
        : undefined,
    }
    return c.json(resultJSON)
  })

  return app
}
