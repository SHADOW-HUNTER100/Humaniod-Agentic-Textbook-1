import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'AI-Native Software Development & Physical AI',
  tagline: 'Research and Development in Humanoid Robotics and Physical AI Systems',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://humaniod-agentic-textbook-1.vercel.app', // Vercel deployment URL
  // Set the /<baseUrl>/ pathname under which your site is served
  // For Vercel deployment, use '/' for root or adjust as needed
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'SHADOW-HUNTER100', // Usually your GitHub org/user name.
  projectName: 'Humaniod-Agentic-Textbook-1', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'es', 'fr', 'de', 'it', 'pt', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      es: {
        label: 'Español',
        direction: 'ltr',
      },
      fr: {
        label: 'Français',
        direction: 'ltr',
      },
      de: {
        label: 'Deutsch',
        direction: 'ltr',
      },
      it: {
        label: 'Italiano',
        direction: 'ltr',
      },
      pt: {
        label: 'Português',
        direction: 'ltr',
      },
      ur: {
        label: 'اردو',
        direction: 'rtl',
      },
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SHADOW-HUNTER100/Humaniod-Agentic-Textbook-1/tree/main/Textbook/docs',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SHADOW-HUNTER100/Humaniod-Agentic-Textbook-1/tree/main/Textbook/docs',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    scripts: [
      {
        src: '/js/translation.js',
        async: true,
      },
      {
        src: '/js/animations.js',
        async: true,
      },
    ],
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'doc',
          docId: 'weekly-breakdown',
          position: 'left',
          label: 'Weekly Breakdown',
        },
        {
          type: 'doc',
          docId: 'assessments',
          position: 'left',
          label: 'Assessments',
        },
        {
          href: 'https://github.com/SHADOW-HUNTER100/Humaniod-Agentic-Textbook-1',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro',
            },
            {
              label: 'Physical AI Concepts',
              to: '/docs/physical-ai/concepts',
            },
            {
              label: 'ROS 2 Integration',
              to: '/docs/ros2/integration',
            },
            {
              label: 'Simulation Framework',
              to: '/docs/simulation/framework',
            },
          ],
        },
        {
          title: 'Research Areas',
          items: [
            {
              label: 'Humanoid Robotics',
              to: '/docs/robotics/humanoid',
            },
            {
              label: 'Vision-Language-Action',
              to: '/docs/vla/systems',
            },
            {
              label: 'Digital Twins',
              to: '/docs/simulation/digital-twins',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/SHADOW-HUNTER100/Humaniod-Agentic-Textbook-1',
            },
            {
              label: 'NVIDIA Isaac',
              href: 'https://developer.nvidia.com/isaac',
            },
            {
              label: 'ROS 2 Documentation',
              href: 'https://docs.ros.org/en/humble/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI-Native Software Development & Physical AI Project. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};


export default config;
