import type { ChatStreamPayload, UIChatMessage } from '@lobechat/types';
import { estimateTokenCount } from 'tokenx';

import {
  chatHistoryPrompts,
  compressContextSystemPrompt,
  compressContextUserPrompt,
} from '../prompts';

/**
 * `tokenx` under-counts provider tokenizers by roughly 10–15%; match the
 * context-engine compression check (`DEFAULT_DRIFT_MULTIPLIER`) so the fitted
 * prompt stays under the window the provider actually measures.
 */
const TOKEN_DRIFT_MULTIPLIER = 1.25;

const OMITTED_MESSAGE_CONTENT = '\n...[truncated to fit the compression model context window]';

export interface CompressContextOptions {
  /**
   * Maximum input tokens the compression request may use. Older history that
   * does not fit is left out of the prompt instead of sending a request the
   * model rejects (or silently truncates). Omit to send the full history.
   */
  maxInputTokens?: number;
}

export interface FitCompressContextResult {
  /** Existing summary, clipped when it alone would take most of the budget */
  existingSummary?: string;
  messages: UIChatMessage[];
  /** Number of oldest messages left out of the prompt */
  omittedCount: number;
}

/**
 * Share of the budget an existing summary may take, so a runaway summary can
 * never leave no room for the history it is meant to extend.
 */
const MAX_SUMMARY_BUDGET_RATIO = 0.5;

const estimateTokens = (text: string) =>
  Math.ceil(estimateTokenCount(text) * TOKEN_DRIFT_MULTIPLIER);

/**
 * Clip `text` to roughly `maxTokens`, keeping the head so the opening
 * (command, request, file path) survives. Characters are scaled by the token
 * share with a 10% margin for uneven token density.
 */
const clipToTokens = (text: string, tokens: number, maxTokens: number) =>
  text.slice(0, Math.max(0, Math.floor(text.length * (maxTokens / tokens) * 0.9))) +
  OMITTED_MESSAGE_CONTENT;

const formatHistory = (messages: UIChatMessage[], omittedCount: number) => {
  const omittedNote =
    omittedCount > 0
      ? `Note: ${omittedCount} earlier messages were omitted because the full history exceeds the compression model context window. Summarize the history below.\n\n`
      : '';

  return `${omittedNote}${chatHistoryPrompts(messages)}`;
};

const buildUserContent = (history: string, existingSummary?: string) =>
  `${existingSummary ? `Existing conversation summary:\n${existingSummary}\n\nNew conversation history:\n` : ''}${history}

${compressContextUserPrompt}`;

/**
 * Keep the newest messages that fit `maxInputTokens`, dropping the oldest
 * first. The newest messages carry the work in progress, and anything older
 * than the existing summary's coverage is already the least relevant. When the
 * newest message alone exceeds the budget its content is clipped so the
 * request still fits.
 */
export const fitCompressContextMessages = (
  messages: UIChatMessage[],
  existingSummary: string | undefined,
  maxInputTokens: number,
): FitCompressContextResult => {
  const maxSummaryTokens = Math.floor(maxInputTokens * MAX_SUMMARY_BUDGET_RATIO);
  const summaryTokens = existingSummary ? estimateTokens(existingSummary) : 0;
  const summary =
    existingSummary && summaryTokens > maxSummaryTokens
      ? clipToTokens(existingSummary, summaryTokens, maxSummaryTokens)
      : existingSummary;

  // Reserve room for the fixed prompt parts plus the omitted-messages note.
  const fixedTokens =
    estimateTokens(compressContextSystemPrompt) +
    estimateTokens(buildUserContent(formatHistory([], messages.length), summary));
  let remaining = maxInputTokens - fixedTokens;

  const kept: UIChatMessage[] = [];

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    // Matches one `<role>content</role>` line in `chatHistoryPrompts`.
    const tokens = estimateTokens(`<${message.role}>${message.content}</${message.role}>\n`);

    if (tokens <= remaining) {
      kept.unshift(message);
      remaining -= tokens;
      continue;
    }

    if (kept.length === 0 && remaining > 0) {
      const content = typeof message.content === 'string' ? message.content : '';
      kept.unshift({ ...message, content: clipToTokens(content, tokens, remaining) });
    }

    break;
  }

  return { existingSummary: summary, messages: kept, omittedCount: messages.length - kept.length };
};

/**
 * Chain for compressing conversation context into a summary
 * Used when conversation history exceeds token threshold
 */
export const chainCompressContext = (
  messages: UIChatMessage[],
  existingSummary?: string,
  options: CompressContextOptions = {},
): Partial<ChatStreamPayload> => {
  const fitted =
    options.maxInputTokens && options.maxInputTokens > 0
      ? fitCompressContextMessages(messages, existingSummary, options.maxInputTokens)
      : { existingSummary, messages, omittedCount: 0 };

  return {
    messages: [
      {
        content: compressContextSystemPrompt,
        role: 'system',
      },
      {
        content: buildUserContent(
          formatHistory(fitted.messages, fitted.omittedCount),
          fitted.existingSummary,
        ),
        role: 'user',
      },
    ],
  };
};
