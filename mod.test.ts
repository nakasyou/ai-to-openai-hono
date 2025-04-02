import { assertEquals } from '@std/assert'
import type { LanguageModelV1 } from '@ai-sdk/provider'
import { createOpenAICompat } from './mod.ts'
import OpenAI, {} from 'openai'
import type { Fetch } from 'openai/core'

Deno.test('Works simple conversation', async () => {
  const USAGE = {
    completionTokens: 485144785,
    promptTokens: 54784,
    totalTokens: 485144785 + 54784,
  }
  const TEXT = 'Hello, how are you?'
  const mockModel: LanguageModelV1 = {
    modelId: 'test',
    provider: 'test',
    specificationVersion: 'v1',
    defaultObjectGenerationMode: 'tool',
    doGenerate() {
      return Promise.resolve({
        rawCall: {
          rawPrompt: '',
          rawSettings: {},
        },
        finishReason: 'stop',
        usage: USAGE,
        text: TEXT,
      })
    },
    doStream() {
      return Promise.resolve({
        rawCall: {
          rawPrompt: '',
          rawSettings: {},
        },
        stream: new ReadableStream({}),
      })
    },
  }

  const app = createOpenAICompat({
    languageModels: {
      test: mockModel,
    },
  })

  const client = new OpenAI({
    fetch: (async (info: RequestInfo | URL, init: RequestInit) => {
      const req = new Request(info, init)
      return await app.fetch(req)
    }) as unknown as Fetch,
    apiKey: 'test',
  })

  const response = await client.chat.completions.create({
    model: 'test',
    messages: [{
      role: 'user',
      content: 'Hello',
    }],
  })

  assertEquals(response.usage, {
    completion_tokens: USAGE.completionTokens,
    prompt_tokens: USAGE.promptTokens,
    total_tokens: USAGE.totalTokens,
  })
  assertEquals(response.model, 'test')
  assertEquals(response.choices[0].finish_reason, 'stop')
  assertEquals(response.choices[0].message.role, 'assistant')
  assertEquals(response.choices[0].message.content, TEXT)
})

Deno.test('Streaming', async () => {
  const USAGE = {
    completionTokens: 485144785,
    promptTokens: 54784,
    totalTokens: 485144785 + 54784,
  }
  const CHUNKS = ['Hello', ', ', 'how ', 'ar', 'e ', 'you', '?']
  const mockModel: LanguageModelV1 = {
    modelId: 'test',
    provider: 'test',
    specificationVersion: 'v1',
    defaultObjectGenerationMode: 'tool',
    doGenerate() {
      return Promise.resolve({
        rawCall: {
          rawPrompt: '',
          rawSettings: {},
        },
        finishReason: 'stop',
        usage: USAGE,
      })
    },
    doStream() {
      return Promise.resolve({
        rawCall: {
          rawPrompt: '',
          rawSettings: {},
        },
        stream: new ReadableStream({
          start(controller) {
            for (const chunk of CHUNKS) {
              controller.enqueue({
                type: 'text-delta',
                textDelta: chunk,
              })
            }
            controller.enqueue({
              type: 'finish',
              finishReason: 'stop',
              usage: USAGE,
            })
            controller.close()
          },
        }),
      })
    },
  }

  const app = createOpenAICompat({
    languageModels: {
      test: mockModel,
    },
  })

  const client = new OpenAI({
    fetch: (async (info: RequestInfo | URL, init: RequestInit) => {
      const req = new Request(info, init)
      return await app.fetch(req)
    }) as unknown as Fetch,
    apiKey: 'test',
  })

  const response = await client.chat.completions.create({
    model: 'test',
    messages: [{
      role: 'user',
      content: 'Hello',
    }],
    stream: true,
  })

  const responses = await Array.fromAsync(response)

  assertEquals(
    responses.slice(0, -1).map((r) => r.choices[0].delta.content),
    CHUNKS,
  )
  assertEquals(responses.at(-1)?.choices[0].finish_reason, 'stop')
  assertEquals(responses.at(-1)?.usage, {
    completion_tokens: USAGE.completionTokens,
    total_tokens: USAGE.totalTokens,
    prompt_tokens: USAGE.promptTokens,
  })
})

Deno.test('Verify API key', async () => {
  const app = createOpenAICompat({
    verifyAPIKey: (key) => key === 'test',
    languageModels: {},
  })

  assertEquals(
    (await app.request('/v1/chat/completions', {
      method: 'POST',
    })).status,
    403,
  )
  assertEquals(
    (await app.request('/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer test',
      },
      body: `{}`,
    })).status,
    400, // model not found
  )
  assertEquals(
    (await app.request('/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer test2',
      },
    })).status,
    403,
  )
})
