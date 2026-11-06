import type {
  CompressionGroupCreateInput,
  CompressionGroupCreateResult,
  CompressionGroupFinalizeInput,
  CompressionGroupFinalizeResult,
  CompressionGroupRollbackInput,
  CompressionGroupRollbackResult,
  CompressionPromptInput,
  CompressionPromptResult,
  CompressionTransport,
} from '@lobechat/agent-runtime';
import { getModelPropertyWithFallback } from '@lobechat/model-runtime';
import { chainCompressContext } from '@lobechat/prompts';

import type { LobeChatDatabase } from '@/database/type';
import { MessageService } from '@/server/services/message';

/**
 * Window assumed when the compression model has no `contextWindowTokens`;
 * matches the agent runtime's compression-threshold default.
 */
const DEFAULT_COMPRESSION_WINDOW_TOKENS = 128_000;

/**
 * Share of the compression model's window the prompt may use. The rest is left
 * for the generated summary.
 */
const COMPRESSION_INPUT_WINDOW_RATIO = 0.75;

/**
 * Server {@link CompressionTransport} adapter — owns DB-backed compression
 * group persistence and the compression prompt policy used by the server path.
 */
export class ServerCompressionTransport implements CompressionTransport {
  constructor(
    private readonly serverDB: LobeChatDatabase,
    private readonly userId: string,
    private readonly defaultWorkspaceId?: string,
  ) {}

  /**
   * The history to summarize is every uncompressed message in the topic, which
   * can be far larger than the context that triggered compression (a long-lived
   * topic reached 1.7M tokens against a 1M window). Fit it to the compression
   * model's window so the request is not rejected on every step.
   */
  async buildPrompt(input: CompressionPromptInput): Promise<CompressionPromptResult> {
    // A model-catalog failure must not fail compression; the default window
    // still keeps the request bounded.
    const contextWindowTokens = input.model
      ? await getModelPropertyWithFallback<number | undefined>(
          input.model,
          'contextWindowTokens',
          input.provider,
        ).catch(() => undefined)
      : undefined;
    const maxInputTokens = Math.floor(
      (contextWindowTokens || DEFAULT_COMPRESSION_WINDOW_TOKENS) * COMPRESSION_INPUT_WINDOW_RATIO,
    );

    const payload = chainCompressContext(input.messages, input.existingSummary, {
      maxInputTokens,
    });
    return { messages: payload.messages! };
  }

  async createGroup(input: CompressionGroupCreateInput): Promise<CompressionGroupCreateResult> {
    const service = this.createService(input.workspaceId);
    const result = await service.createCompressionGroup(input.topicId, input.messageIds, {
      agentId: input.agentId,
      groupId: input.groupId,
      threadId: input.threadId,
      topicId: input.topicId,
    } as any);

    return {
      messageGroupId: result.messageGroupId,
      messages: result.messages,
      messagesToSummarize: result.messagesToSummarize,
    };
  }

  async finalizeGroup(
    input: CompressionGroupFinalizeInput,
  ): Promise<CompressionGroupFinalizeResult> {
    const service = this.createService(input.workspaceId);
    const result = await service.finalizeCompression(input.messageGroupId, input.content, {
      agentId: input.agentId,
      groupId: input.groupId,
      sourceGroupIds: input.sourceGroupIds,
      threadId: input.threadId,
      topicId: input.topicId,
    } as any);

    return {
      messages: result.messages,
    };
  }

  async rollbackGroup(
    input: CompressionGroupRollbackInput,
  ): Promise<CompressionGroupRollbackResult> {
    const service = this.createService(input.workspaceId);
    const result = await service.cancelCompression(input.messageGroupId, {
      agentId: input.agentId,
      groupId: input.groupId,
      threadId: input.threadId,
      topicId: input.topicId,
    } as any);

    return { messages: result.messages };
  }

  private createService(workspaceId?: string) {
    return new MessageService(this.serverDB, this.userId, workspaceId ?? this.defaultWorkspaceId);
  }
}
