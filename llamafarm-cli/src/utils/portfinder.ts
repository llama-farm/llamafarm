import * as net from 'net';

export async function getPort(startPort: number = 8080): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    
    server.listen(startPort, () => {
      const port = (server.address() as net.AddressInfo).port;
      server.close(() => resolve(port));
    });
    
    server.on('error', (err: any) => {
      if (err.code === 'EADDRINUSE') {
        // Port is in use, try the next one
        resolve(getPort(startPort + 1));
      } else {
        reject(err);
      }
    });
  });
}