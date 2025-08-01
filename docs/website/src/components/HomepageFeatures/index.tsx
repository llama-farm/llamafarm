import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  emoji: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Configuration as Code',
    emoji: '📝',
    description: (
      <>
        Define your entire AI infrastructure in simple YAML files. 
        Version control your models, prompts, and deployment configurations. 
        No more scattered scripts or manual setups.
      </>
    ),
  },
  {
    title: 'True Model Portability',
    emoji: '🔄',
    description: (
      <>
        Switch between Llama, GPT-4, Claude, or any model with a single config change. 
        Run locally for development, deploy to cloud for production. 
        Your choice, your control.
      </>
    ),
  },
  {
    title: 'Privacy by Default',
    emoji: '🔒',
    description: (
      <>
        Keep your data where it belongs - with you. Run models locally, 
        process sensitive data on-premises, and only use cloud when you choose to. 
        Complete data sovereignty.
      </>
    ),
  },
];

function Feature({title, emoji, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <div className={styles.featureEmoji}>{emoji}</div>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}