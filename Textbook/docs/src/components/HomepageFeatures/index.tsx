import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
  to?: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: Robotic Nervous System (ROS 2)',
    description: (
      <>
        Learn ROS 2 fundamentals: Nodes, Topics, Services. Bridge Python Agents to ROS controllers using rclpy. Understand URDF for humanoid robots.
      </>
    ),
    to: '/docs/modules/module-1-ros2-nervous-system',
  },
  {
    title: 'Module 2: Digital Twin Simulation',
    description: (
      <>
        Master physics simulation with Gazebo and Unity. Simulate sensors like LiDAR and cameras. High-fidelity rendering for human-robot interaction.
      </>
    ),
    to: '/docs/modules/module-2-digital-twin-simulation',
  },
  {
    title: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
    description: (
      <>
        Advanced perception with NVIDIA Isaac Sim. Hardware-accelerated VSLAM and navigation. Path planning for bipedal humanoid movement.
      </>
    ),
    to: '/docs/modules/module-3-ai-robot-brain',
  },
  {
    title: 'Module 4: Vision-Language-Action (VLA)',
    description: (
      <>
        Convergence of LLMs and Robotics. Voice commands with OpenAI Whisper. Cognitive planning from natural language to ROS actions.
      </>
    ),
    to: '/docs/modules/module-4-vla-integration',
  },
];

function Feature({title, description, to}: FeatureItem) {
  const featureContent = (
    <div className={styles.featureCard}>
      <div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );

  return (
    <div className={clsx('col col--3')}>
      {to ? (
        <Link to={to} style={{textDecoration: 'none', display: 'block'}}>
          {featureContent}
        </Link>
      ) : (
        featureContent
      )}
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
