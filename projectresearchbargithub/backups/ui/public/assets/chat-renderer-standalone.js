
/**
 * Standalone Chat Renderer Bundle
 * Generated from TypeScript sources for WebKit compatibility
 */

// Validation Module (inlined)
/**
 * Runtime validation for Python→JavaScript data transfer
 */
class ValidationError extends Error {
    constructor(message, path = '') {
        super(`Validation Error${path ? ` at ${path}` : ''}: ${message}`);
        this.path = path;
        this.name = 'ValidationError';
    }
}
/**
 * Type guards for runtime validation
 */
const Guards = {
    isString(value) {
        return typeof value === 'string';
    },
    isNumber(value) {
        return typeof value === 'number' && !isNaN(value);
    },
    isBoolean(value) {
        return typeof value === 'boolean';
    },
    isMessageRole(value) {
        return this.isString(value) && ['user', 'assistant', 'system'].includes(value);
    },
    isOptionalString(value) {
        return value === undefined || this.isString(value);
    },
    isChatMessage(value) {
        if (typeof value !== 'object' || value === null) {
            return false;
        }
        const obj = value;
        return (this.isMessageRole(obj.role) &&
            this.isString(obj.content) &&
            this.isOptionalString(obj.timestamp) &&
            this.isOptionalString(obj.id));
    },
    isChatData(value) {
        if (typeof value !== 'object' || value === null) {
            return false;
        }
        const obj = value;
        return (this.isString(obj.id) &&
            this.isString(obj.title) &&
            Array.isArray(obj.messages) &&
            obj.messages.every(msg => this.isChatMessage(msg)) &&
            this.isString(obj.created) &&
            this.isString(obj.updated) &&
            this.isOptionalString(obj.model));
    }
};
/**
 * Validators that throw ValidationError on invalid data
 */
const Validators = {
    validateMessageRole(value, path = 'role') {
        if (!Guards.isMessageRole(value)) {
            throw new ValidationError(`Expected one of 'user', 'assistant', 'system', got ${typeof value}: ${JSON.stringify(value)}`, path);
        }
        return value;
    },
    validateString(value, path = 'string') {
        if (!Guards.isString(value)) {
            throw new ValidationError(`Expected string, got ${typeof value}: ${JSON.stringify(value)}`, path);
        }
        return value;
    },
    validateOptionalString(value, path = 'optionalString') {
        if (value === undefined) {
            return undefined;
        }
        return this.validateString(value, path);
    },
    validateBoolean(value, path = 'boolean') {
        if (!Guards.isBoolean(value)) {
            throw new ValidationError(`Expected boolean, got ${typeof value}: ${JSON.stringify(value)}`, path);
        }
        return value;
    },
    validateChatMessage(value, path = 'message') {
        if (!Guards.isChatMessage(value)) {
            throw new ValidationError(`Invalid chat message structure: ${JSON.stringify(value)}`, path);
        }
        return value;
    },
    validateChatData(value, path = 'chatData') {
        if (!Guards.isChatData(value)) {
            throw new ValidationError(`Invalid chat data structure: ${JSON.stringify(value)}`, path);
        }
        return value;
    },
    validateAddMessageParams(role, content) {
        return {
            role: this.validateMessageRole(role, 'addMessage.role'),
            content: this.validateString(content, 'addMessage.content')
        };
    },
    validateUpdateStreamingParams(role, content, isComplete = false) {
        return {
            role: this.validateMessageRole(role, 'updateStreaming.role'),
            content: this.validateString(content, 'updateStreaming.content'),
            isComplete: this.validateBoolean(isComplete, 'updateStreaming.isComplete')
        };
    },
    validateShowLoadingParams(show) {
        return {
            show: this.validateBoolean(show, 'showLoading.show')
        };
    }
};
/**
 * Safe wrappers for Python bridge functions
 */
function createSafeWrapper(fn, validator, functionName) {
    return (...args) => {
        try {
            const validatedArgs = validator(...args);
            return fn(...validatedArgs);
        }
        catch (error) {
            if (error instanceof ValidationError) {
                console.error(`${functionName}: ${error.message}`);
                console.error('Received arguments:', args);
            }
            else {
                console.error(`${functionName}: Unexpected error`, error);
            }
            // Don't throw in production to prevent WebView crashes
            return;
        }
    };
}


// Chat Renderer Module (inlined)  
/**
 * Chat Renderer - TypeScript implementation
 * Extracted from chat_template.html for better type safety and maintainability
 */

class ChatRenderer {
    constructor() {
        this.messagesContainer = document.getElementById('messages');
        this.loadingElement = document.getElementById('loading');
        if (!this.messagesContainer || !this.loadingElement) {
            throw new Error('Required DOM elements not found');
        }
    }
    /**
     * Escape HTML characters to prevent XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        };
        return text.replace(/[&<>"']/g, (match) => map[match] || match);
    }
    /**
     * Process markdown-like formatting
     */
    processMarkdown(text) {
        return text
            .replace(/^###\s+(.+)$/gm, '<span class="h3">$1</span>')
            .replace(/^##\s+(.+)$/gm, '<span class="h2">$1</span>')
            .replace(/^#\s+(.+)$/gm, '<span class="h1">$1</span>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    }
    /**
     * Process text with inline code, code blocks, and markdown
     */
    processText(text) {
        const inlineCode = [];
        const codeBlocks = [];
        let inlineIndex = 0;
        let blockIndex = 0;
        // Extract inline code first
        let processedText = text.replace(/`([^`]+)`/g, (_, code) => {
            inlineCode[inlineIndex] = `<code>${this.escapeHtml(code)}</code>`;
            return `__INLINE_${inlineIndex++}__`;
        });
        // Extract code blocks
        processedText = processedText.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, body) => {
            const language = lang || 'text';
            const escapedBody = this.escapeHtml(body);
            codeBlocks[blockIndex] = this.createCodeBlock(language, body, escapedBody);
            return `__CODE_${blockIndex++}__`;
        });
        // Process markdown
        processedText = this.processMarkdown(processedText);
        // Restore inline code
        processedText = processedText.replace(/__INLINE_(\d+)__/g, (_, index) => {
            return inlineCode[parseInt(index, 10)] || '';
        });
        // Restore code blocks
        processedText = processedText.replace(/__CODE_(\d+)__/g, (_, index) => {
            return codeBlocks[parseInt(index, 10)] || '';
        });
        return processedText;
    }
    /**
     * Create a code block with enhanced UI
     */
    createCodeBlock(language, rawContent, escapedContent) {
        const languageLabel = language || 'text';
        const dataAttr = this.escapeHtml(rawContent).replace(/"/g, '&quot;');
        return `
      <div class="code-block-container">
        <div class="code-block-header">
          <span class="language-label">${languageLabel}</span>
          <div class="copy-buttons">
            <button class="copy-btn" onclick="chatRenderer.copyCodeBlock(this, 'raw')">Copy</button>
            <button class="copy-btn" onclick="chatRenderer.copyCodeBlock(this, 'formatted')">Copy Formatted</button>
          </div>
        </div>
        <pre><code class="language-${language}" data-raw-content="${dataAttr}">${escapedContent}</code></pre>
      </div>
    `;
    }
    /**
     * Copy code block with different format options
     */
    copyCodeBlock(button, format) {
        const container = button.closest('.code-block-container');
        if (!container) {
            console.error('Code block container not found');
            return;
        }
        const codeElement = container.querySelector('code');
        const languageLabel = container.querySelector('.language-label');
        if (!codeElement || !languageLabel) {
            console.error('Required code block elements not found');
            return;
        }
        const rawContent = codeElement.getAttribute('data-raw-content') || codeElement.textContent || '';
        const language = languageLabel.textContent || 'text';
        let textToCopy;
        switch (format) {
            case 'raw':
                textToCopy = rawContent;
                break;
            case 'formatted':
                textToCopy = `\`\`\`${language}\n${rawContent}\n\`\`\``;
                break;
            case 'with-line-numbers':
                const lines = rawContent.split('\n');
                const numberedLines = lines.map((line, index) => `${(index + 1).toString().padStart(3)}: ${line}`);
                textToCopy = numberedLines.join('\n');
                break;
            default:
                textToCopy = rawContent;
        }
        navigator.clipboard.writeText(textToCopy)
            .then(() => {
            this.showCopyFeedback(button, 'Copied!', '#98971a');
        })
            .catch((err) => {
            console.error('Failed to copy:', err);
            this.showCopyFeedback(button, 'Failed', '#cc241d');
        });
    }
    /**
     * Show visual feedback for copy operations
     */
    showCopyFeedback(button, message, color) {
        const originalText = button.textContent || '';
        const originalColor = button.style.backgroundColor;
        button.textContent = message;
        button.style.backgroundColor = color;
        setTimeout(() => {
            button.textContent = originalText;
            button.style.backgroundColor = originalColor;
        }, 2000);
    }
    /**
     * Render math equations outside of code blocks
     */
    renderMathInExceptCodeBlocks(element) {
        if (!window.renderMathInElement) {
            // KaTeX not loaded, retry later
            setTimeout(() => this.renderMathInExceptCodeBlocks(element), 25);
            return;
        }
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
            acceptNode: (node) => {
                // Skip text nodes inside code blocks
                let parent = node.parentElement;
                while (parent && parent !== element) {
                    if (parent.tagName === 'CODE' ||
                        parent.tagName === 'PRE' ||
                        parent.classList.contains('code-block-container')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    parent = parent.parentElement;
                }
                return NodeFilter.FILTER_ACCEPT;
            }
        });
        const textNodes = [];
        let node;
        while ((node = walker.nextNode())) {
            textNodes.push(node);
        }
        // Apply math rendering only to text nodes outside code blocks
        textNodes.forEach((textNode) => {
            const content = textNode.textContent || '';
            if (content.includes('$') || content.includes('\\(') || content.includes('\\[')) {
                const parent = textNode.parentElement;
                if (parent && window.renderMathInElement) {
                    const macros = {
                        "\\RR": "\\mathbb{R}",
                        "\\ZZ": "\\mathbb{Z}",
                        "\\NN": "\\mathbb{N}",
                        "\\QQ": "\\mathbb{Q}",
                        "\\CC": "\\mathbb{C}",
                        "\\PP": "\\mathbb{P}",
                        "\\HH": "\\mathbb{H}",
                        "\\eps": "\\varepsilon",
                        "\\normalout": "\\mathrm{normal}_{\\mathrm{out}}",
                        "\\normalin": "\\mathrm{normal}_{\\mathrm{in}}",
                        "\\vol": "\\operatorname{vol}",
                        "\\dist": "\\operatorname{dist}"
                    };
                    window.renderMathInElement(parent, {
                        delimiters: [
                            { left: '$$', right: '$$', display: true },
                            { left: '\\[', right: '\\]', display: true },
                            { left: '$', right: '$', display: false },
                            { left: '\\(', right: '\\)', display: false }
                        ],
                        throwOnError: false,
                        macros: macros
                    });
                }
            }
        });
    }
    /**
     * Create a message bubble
     */
    createBubble(role, text) {
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = this.processText(text);
        // Render math only outside code blocks
        this.renderMathInExceptCodeBlocks(bubble);
        // Add copy functionality for any legacy code blocks
        this.addLegacyCopyFunctionality(bubble);
        return bubble;
    }
    /**
     * Add copy functionality to legacy code blocks (without the new structure)
     */
    addLegacyCopyFunctionality(scope) {
        scope.querySelectorAll('pre code').forEach((block) => {
            const container = block.closest('.code-block-container');
            if (container)
                return; // Skip new-style code blocks
            const pre = block.parentElement;
            if (!pre || pre.querySelector('.code-block-header'))
                return;
            const header = document.createElement('div');
            header.className = 'code-block-header';
            const langClass = Array.from(block.classList).find(c => c.startsWith('language-'));
            if (langClass) {
                const lang = langClass.replace('language-', '');
                const label = document.createElement('span');
                label.className = 'language-label';
                label.textContent = lang;
                header.appendChild(label);
            }
            const btn = document.createElement('button');
            btn.className = 'copy-btn';
            btn.textContent = 'Copy';
            btn.onclick = () => this.copyCodeBlock(btn, 'raw');
            header.appendChild(btn);
            pre.insertBefore(header, block);
        });
    }
    /**
     * Add a message to the chat
     */
    addMessage(role, text) {
        const message = document.createElement('div');
        message.className = `message ${role}`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';
        const content = document.createElement('div');
        content.className = 'content';
        content.appendChild(this.createBubble(role, text));
        if (role === 'user') {
            message.append(content, avatar);
        }
        else {
            message.append(avatar, content);
        }
        this.messagesContainer.appendChild(message);
        this.scrollToBottom();
    }
    /**
     * Update streaming message
     */
    updateStreamingMessage(role, text, isComplete = false) {
        const lastMessage = this.messagesContainer.lastElementChild;
        if (!lastMessage || !lastMessage.classList.contains(role)) {
            this.addMessage(role, text);
            return;
        }
        const bubble = lastMessage.querySelector('.bubble');
        if (!bubble)
            return;
        if (isComplete) {
            bubble.innerHTML = this.processText(text);
            this.renderMathInExceptCodeBlocks(bubble);
            this.addLegacyCopyFunctionality(bubble);
        }
        else {
            bubble.textContent = text;
        }
        this.scrollToBottom();
    }
    /**
     * Clear all messages
     */
    clearMessages() {
        this.messagesContainer.innerHTML = '';
    }
    /**
     * Show/hide loading indicator
     */
    showLoading(show) {
        this.loadingElement.style.display = show ? 'block' : 'none';
    }
    /**
     * Scroll to bottom of chat
     */
    scrollToBottom() {
        try {
            window.scroll({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
        catch {
            window.scrollTo(0, document.body.scrollHeight);
        }
    }
}
// Global instance
const chatRenderer = new ChatRenderer();
// Create safe wrappers with validation for Python bridge
const safeAddMessage = createSafeWrapper(chatRenderer.addMessage.bind(chatRenderer), (role, content) => {
    const validated = Validators.validateAddMessageParams(role, content);
    return [validated.role, validated.content];
}, 'addMessage');
const safeUpdateStreamingMessage = createSafeWrapper(chatRenderer.updateStreamingMessage.bind(chatRenderer), (role, content, isComplete = false) => {
    const validated = Validators.validateUpdateStreamingParams(role, content, isComplete);
    return [validated.role, validated.content, validated.isComplete];
}, 'updateStreamingMessage');
const safeShowLoading = createSafeWrapper(chatRenderer.showLoading.bind(chatRenderer), (show) => {
    const validated = Validators.validateShowLoadingParams(show);
    return [validated.show];
}, 'showLoading');
// Expose to global scope for Python bridge with validation
window.addMessage = safeAddMessage;
window.updateStreamingMessage = safeUpdateStreamingMessage;
window.clearMessages = chatRenderer.clearMessages.bind(chatRenderer);
window.showLoading = safeShowLoading;
window.chatRenderer = chatRenderer;
// Mark as ready
window.webviewReady = true;
