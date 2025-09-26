import Head from '@docusaurus/Head'

export default function RootMetaRedirect() {
  return (
    <Head>
      <meta httpEquiv="refresh" content="0; url=/docs/intro" />
      <meta name="robots" content="noindex" />
    </Head>
  )
}
