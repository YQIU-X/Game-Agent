class WebSocketService {
  constructor() {
    this.ws = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`ws://localhost:3000`);
        
        this.ws.onopen = () => {
          console.log('[WebSocket] 连接成功');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('[WebSocket] 消息解析错误:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('[WebSocket] 连接关闭');
          this.isConnected = false;
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('[WebSocket] 连接错误:', error);
          this.isConnected = false;
          reject(error);
        };

      } catch (error) {
        console.error('[WebSocket] 连接失败:', error);
        reject(error);
      }
    });
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`[WebSocket] 尝试重连 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      
      setTimeout(() => {
        this.connect().catch(() => {
          this.attemptReconnect();
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('[WebSocket] 重连失败，已达到最大重试次数');
    }
  }

  send(message) {
    if (this.isConnected && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('[WebSocket] 发送消息失败:', error);
      }
    } else {
      console.warn('[WebSocket] 连接未建立，无法发送消息');
    }
  }

  sendPlayerAction(processId, action) {
    this.send({
      type: 'player_action',
      processId,
      action
    });
  }

  handleMessage(data) {
    console.log('[WebSocket] 收到消息:', data);
    
    // 触发对应的事件监听器
    if (this.listeners.has(data.type)) {
      this.listeners.get(data.type).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('[WebSocket] 事件处理错误:', error);
        }
      });
    } else {
      console.log('[WebSocket] 没有找到事件监听器:', data.type);
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.listeners.clear();
  }
}

// 创建单例实例
const websocketService = new WebSocketService();

export default websocketService;
