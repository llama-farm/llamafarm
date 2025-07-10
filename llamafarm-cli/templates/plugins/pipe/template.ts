import { BasePipePlugin } from '../../base';

export class {{NAME}}PipePlugin extends BasePipePlugin {
  name = '{{LOWERCASE_NAME}}-pipe';
  version = '1.0.0';
  protocol = '{{PROTOCOL}}'; // ws, http, webrtc, sse
  description = '{{DESCRIPTION}}';
  author = '{{AUTHOR}}';
  dependencies = []; // npm packages needed

  createServer(options: any): any {
    const { port = 8080, ...config } = options;
    
    console.log(`🌐 Creating ${this.protocol} server on port ${port}`);
    
    // Create and return your server instance
    // Handle connections, messages, etc.
  }

  createClient(options: any): any {
    const { url, ...config } = options;
    
    console.log(`🔌 Creating ${this.protocol} client for ${url}`);
    
    // Create and return your client instance
    // Handle connection, messages, etc.
  }
}

export default new {{NAME}}PipePlugin();
