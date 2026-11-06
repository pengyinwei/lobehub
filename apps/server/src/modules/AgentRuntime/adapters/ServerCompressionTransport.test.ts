import { getModelPropertyWithFallback } from '@lobechat/model-runtime';
import { chainCompressContext } from '@lobechat/prompts';
import type { UIChatMessage } from '@lobechat/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ServerCompressionTransport } from './ServerCompressionTransport';

vi.mock('@lobechat/model-runtime', () => ({
  getModelPropertyWithFallback: vi.fn(),
}));

const toolMessage = (index: number): UIChatMessage =>
  ({
    content: Array.from({ length: 400 }, (_, line) => `msg${index}-line-${line} v=${line}`).join(
      '\n',
    ),
    id: `msg-${index}`,
    role: 'tool',
  }) as UIChatMessage;

describe('ServerCompressionTransport.buildPrompt', () => {
  const transport = new ServerCompressionTransport({} as any, 'user-1');
  // ~100k tokens of history
  const history = Array.from({ length: 50 }, (_, index) => toolMessage(index));

  beforeEach(() => {
    vi.mocked(getModelPropertyWithFallback).mockReset();
  });

  it('fits the history into the compression model context window', async () => {
    vi.mocked(getModelPropertyWithFallback).mockResolvedValue(32_000);

    const { messages } = await transport.buildPrompt({
      messages: history,
      model: 'deepseek-flash',
      provider: 'lobehub',
    });

    expect(getModelPropertyWithFallback).toHaveBeenCalledWith(
      'deepseek-flash',
      'contextWindowTokens',
      'lobehub',
    );
    expect(messages).toEqual(
      chainCompressContext(history, undefined, { maxInputTokens: 24_000 }).messages,
    );
    expect(messages[1].content).toContain('msg49-line-0');
    expect(messages[1].content).not.toContain('msg0-line-0');
  });

  it('falls back to a 128k window when the model has no context window metadata', async () => {
    vi.mocked(getModelPropertyWithFallback).mockResolvedValue(undefined);

    const { messages } = await transport.buildPrompt({
      messages: [...history, ...history, ...history],
      model: 'unknown-model',
      provider: 'custom',
    });

    expect(messages).toEqual(
      chainCompressContext([...history, ...history, ...history], undefined, {
        maxInputTokens: 96_000,
      }).messages,
    );
    expect(messages[1].content).toMatch(/Note: \d+ earlier messages were omitted/);
  });
});
