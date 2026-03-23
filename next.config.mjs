/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['pdf-parse', 'pdfjs-dist', 'mammoth'],
    optimizePackageImports: [
      'lucide-react',
      '@radix-ui/react-dialog',
      '@radix-ui/react-slot',
      '@radix-ui/react-label',
      '@radix-ui/react-tabs',
      '@radix-ui/react-scroll-area',
    ],
  },
};

export default nextConfig;
