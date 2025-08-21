import type { ReactNode } from 'react'
import clsx from 'clsx'
import Link from '@docusaurus/Link'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import useBaseUrl from '@docusaurus/useBaseUrl'
import Layout from '@theme/Layout'
import HomepageFeatures from '@site/src/components/HomepageFeatures'
import Heading from '@theme/Heading'

import styles from './index.module.css'

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext()
  const logoUrl = useBaseUrl('/img/logotype-tall-tan.svg')
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <img src={logoUrl} alt="LlamaFarm" className={styles.heroLogo} />
          {/* Title visually replaced by logotype; keep for SEO but hide visually */}
          <Heading as="h1" className="hero__title sr-only">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <p className={styles.heroDescription}>
            The open-source framework for building, managing, and deploying AI
            applications with any model, on any cloud, anywhere. Take control of
            your AI infrastructure with simple YAML configurations.
          </p>
          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro"
            >
              Get Started 🚀
            </Link>
            <Link
              className="button button--outline button--secondary button--lg"
              to="https://github.com/llama-farm/llamafarm"
            >
              View on GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>
  )
}

function WhyLlamaFarm() {
  return (
    <section className={styles.whySection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Why LlamaFarm?
        </Heading>
        <div className={styles.whyGrid}>
          <div className={styles.whyCard}>
            <h3>🏠 Local First</h3>
            <p>
              Run AI models locally on your hardware. No cloud dependency, no
              data leaving your infrastructure. Complete privacy and control.
            </p>
          </div>
          <div className={styles.whyCard}>
            <h3>🌍 Deploy Anywhere</h3>
            <p>
              One configuration, multiple destinations. Deploy to AWS, Azure,
              GCP, on-premises, or edge devices without changing your code.
            </p>
          </div>
          <div className={styles.whyCard}>
            <h3>🔧 Config-Based</h3>
            <p>
              Define your entire AI pipeline in simple YAML. No complex code,
              just declarative configurations that anyone can understand.
            </p>
          </div>
          <div className={styles.whyCard}>
            <h3>🤖 Any Model</h3>
            <p>
              Works with Llama, GPT, Claude, Mistral, and more. Switch models
              with a single config change. No vendor lock-in.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

function QuickExample() {
  return (
    <section className={styles.exampleSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Simple as YAML
        </Heading>
        <div className={styles.codeExample}>
          <pre>
            <code>{`# llamafarm.yaml
models:
  - name: local-llama
    type: llama2
    path: ./models/llama-2-7b
    
  - name: cloud-gpt
    type: openai
    api_key: $OPENAI_KEY
    
pipeline:
  - input: user_query
  - model: local-llama
    fallback: cloud-gpt
  - output: response
    
deploy:
  targets:
    - local: true
    - aws: 
        region: us-east-1
    - edge:
        devices: ["rpi-cluster"]`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext()
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="LlamaFarm - Config-based AI framework for local and cloud deployment"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <WhyLlamaFarm />
        <QuickExample />
      </main>
    </Layout>
  )
}
