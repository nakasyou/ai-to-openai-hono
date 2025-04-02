# ai-to-openai-hono

An adapter to add OpenAI compatibility to your Hono app using the Vercel AI SDK.

## Install

https://jsr.io/@ns/ai-to-openai-hono

## Usage

```ts
import { Hono } from 'hono'
import { createOpenAIHono } from '@ns/ai-to-openai-hono'

import { anthropic } from 'npm:@ai-sdk/anthropic' // or your favorite provider

const app = new Hono() // Your existing or new Hono app

// Mount the OpenAI-compatible endpoint
app.route(
  '/my-ai-endpoint',
  createOpenAIHono({
    languageModels: {
      'claude-3.7-sonnet': anthropic('claude-3-7-sonnet-20250219'), // Map model names to Vercel AI SDK instances
      // ... add more models here
    },
    // Optional: Add API key verification
    verifyAPIKey(key) {
      // In production, compare against securely stored keys (e.g., environment variables)
      return key === 'this-is-super-secret-key'
    },
  }),
)

export default app
```

Next, run your Hono server. The OpenAI-compatible endpoint will then be
available.

```ts
import { OpenAI } from 'openai'

const openai = new OpenAI({
  baseURL: 'http://localhost:8080/my-ai-endpoint',
  apiKey: 'this-is-super-secret-key',
})

const completion = await openai.chat.completions.create({
  model: 'claude-3.7-sonnet',
  // ...
})
console.log(completion.choices[0].message.content)
```
