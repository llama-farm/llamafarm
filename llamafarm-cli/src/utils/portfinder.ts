import * as net from 'net';

export async function getPort(startPort: number = 8080): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    
    server.listen(startPort, () => {
      const port = (server.address() as net.AddressInfo).port;
      server.close(() => resolve(port));
    });
    
    server.on('error', () => {
      // Port in use, try next one
      getPort(startPort + 1).then(resolve).catch(reject);
    });
  });
}