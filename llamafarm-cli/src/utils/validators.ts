export function validateModelName(model: string): boolean {
  // Basic validation for model names
  const validPattern = /^[a-z0-9]+(-[a-z0-9]+)*$/;
  return validPattern.test(model);
}

export function validateDevice(device: string): boolean {
  const validDevices = ['mac', 'windows', 'linux', 'raspberry-pi', 'jetson'];
  return validDevices.includes(device);
}

export function validateQuantization(quantization: string): boolean {
  const validQuantizations = ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'];
  return validQuantizations.includes(quantization);
} 