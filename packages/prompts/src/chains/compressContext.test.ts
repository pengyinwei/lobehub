import type { UIChatMessage } from '@lobechat/types';
import { estimateTokenCount } from 'tokenx';
import { describe, expect, it } from 'vitest';

import { chainCompressContext, fitCompressContextMessages } from './compressContext';

const message = (id: string, content: string, role = 'tool'): UIChatMessage =>
  ({ content, id, role }) as UIChatMessage;

const promptTokens = (payload: ReturnType<typeof chainCompressContext>) =>
  payload.messages!.reduce(
    (total, item) => total + estimateTokenCount(item.content as string) * 1.25,
    0,
  );

// ~2,000 tokens of varied tool output per message
const largeContent = (seed: string) =>
  Array.from({ length: 400 }, (_, index) => `${seed}-line-${index} value=${index * 7}`).join('\n');

describe('chainCompressContext', () => {
  it('keeps the full history when no input budget is given', () => {
    const messages = [message('m1', 'first', 'user'), message('m2', 'second', 'assistant')];

    const payload = chainCompressContext(messages);

    expect(payload.messages![1].content).toContain('<user>first</user>');
    expect(payload.messages![1].content).toContain('<assistant>second</assistant>');
    expect(payload.messages![1].content).not.toContain('omitted');
  });

  it('fits a history larger than the window by dropping the oldest messages', () => {
    const messages = Array.from({ length: 50 }, (_, index) =>
      message(`m${index}`, largeContent(`msg${index}`)),
    );
    const maxInputTokens = 20_000;

    expect(promptTokens(chainCompressContext(messages))).toBeGreaterThan(maxInputTokens * 3);

    const payload = chainCompressContext(messages, 'prior summary', { maxInputTokens });
    const userContent = payload.messages![1].content as string;

    expect(promptTokens(payload)).toBeLessThanOrEqual(maxInputTokens);
    expect(userContent).toContain('Existing conversation summary:\nprior summary');
    expect(userContent).toContain('msg49-line-0');
    expect(userContent).not.toContain('msg0-line-0');
    expect(userContent).toMatch(/Note: \d+ earlier messages were omitted/);
  });

  it('clips the newest message when it alone exceeds the window', () => {
    const huge = message('m-huge', largeContent('huge').repeat(20));
    const maxInputTokens = 5000;

    const { messages, omittedCount } = fitCompressContextMessages(
      [message('m-old', 'old'), huge],
      undefined,
      maxInputTokens,
    );

    expect(omittedCount).toBe(1);
    expect(messages).toHaveLength(1);
    expect(messages[0].content).toContain('huge-line-0');
    expect(messages[0].content).toContain(
      '[truncated to fit the compression model context window]',
    );
    expect(
      promptTokens(
        chainCompressContext([message('m-old', 'old'), huge], undefined, { maxInputTokens }),
      ),
    ).toBeLessThanOrEqual(maxInputTokens);
  });

  it('clips an oversized existing summary so the newest messages still fit', () => {
    const maxInputTokens = 10_000;
    const hugeSummary = largeContent('summary').repeat(20);

    const result = fitCompressContextMessages(
      [message('m-new', 'latest tool output')],
      hugeSummary,
      maxInputTokens,
    );

    expect(result.messages).toEqual([message('m-new', 'latest tool output')]);
    expect(result.existingSummary).toContain('summary-line-0');
    expect(result.existingSummary!.length).toBeLessThan(hugeSummary.length);
    expect(
      promptTokens(
        chainCompressContext([message('m-new', 'latest tool output')], hugeSummary, {
          maxInputTokens,
        }),
      ),
    ).toBeLessThanOrEqual(maxInputTokens);
  });
});
