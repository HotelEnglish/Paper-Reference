module.exports = {
  async rewrites() {
    return [
      {
        source: '/about',
        destination: '/about.html'
      },
      {
        source: '/blog/:slug',
        destination: '/blog-post?slug=:slug'
      },
      {
        source: '/api/:path*',
        destination: '/api/:path*'
      }
    ];
  },
  async redirects() {
    return [
      {
        source: '/old-url',
        destination: '/new-url',
        permanent: true
      }
    ];
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cache-Control', value: 'public, max-age=0, s-maxage=60' }
        ]
      }
    ];
  },
  trailingSlash: false,
  cleanUrls: true
};
