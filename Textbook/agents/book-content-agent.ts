import fs from 'fs/promises';
import path from 'path';

interface BookContentConfig {
  title: string;
  author: string;
  targetDirectory: string;
  chapters: ChapterConfig[];
}

interface ChapterConfig {
  id: string;
  title: string;
  content: string;
  position: number;
}

class BookContentAgent {
  private config: BookContentConfig;

  constructor(config: BookContentConfig) {
    this.config = config;
  }

  /**
   * Generate new book content and replace old content
   */
  async generateBookContent(): Promise<void> {
    console.log(`Starting to generate book: "${this.config.title}" by ${this.config.author}`);

    // Ensure target directory exists
    await this.ensureTargetDirectory();

    // Remove old content first
    await this.removeOldContent();

    // Generate new content
    await this.generateNewContent();

    // Update sidebar configuration
    await this.updateSidebar();

    console.log(`Successfully generated book: "${this.config.title}"`);
  }

  private async ensureTargetDirectory(): Promise<void> {
    try {
      await fs.access(this.config.targetDirectory);
    } catch {
      await fs.mkdir(this.config.targetDirectory, { recursive: true });
      console.log(`Created target directory: ${this.config.targetDirectory}`);
    }
  }

  private async removeOldContent(): Promise<void> {
    console.log('Removing old content...');

    // Remove all markdown files in the target directory
    const files = await fs.readdir(this.config.targetDirectory);
    const markdownFiles = files.filter(file => path.extname(file) === '.md' || path.extname(file) === '.mdx');

    for (const file of markdownFiles) {
      const filePath = path.join(this.config.targetDirectory, file);
      await fs.unlink(filePath);
      console.log(`Removed old file: ${filePath}`);
    }

    // Remove subdirectories (like specs/*, etc.)
    const subdirs = [];
    for (const file of files) {
      const filePath = path.join(this.config.targetDirectory, file);
      try {
        const stats = await fs.stat(filePath);
        if (stats.isDirectory()) {
          subdirs.push(file);
        }
      } catch (error) {
        console.error(`Error accessing file ${filePath}:`, error);
      }
    }

    for (const subdir of subdirs) {
      const dirPath = path.join(this.config.targetDirectory, subdir);
      await this.removeDirectory(dirPath);
      console.log(`Removed old directory: ${dirPath}`);
    }

    console.log('Finished removing old content.');
  }

  private async removeDirectory(dirPath: string): Promise<void> {
    const entries = await fs.readdir(dirPath);

    for (const entry of entries) {
      const entryPath = path.join(dirPath, entry);
      const stats = await fs.stat(entryPath);

      if (stats.isDirectory()) {
        await this.removeDirectory(entryPath);
      } else {
        await fs.unlink(entryPath);
      }
    }

    await fs.rmdir(dirPath);
  }

  private async generateNewContent(): Promise<void> {
    console.log('Generating new content...');

    // Sort chapters by position
    const sortedChapters = [...this.config.chapters].sort((a, b) => a.position - b.position);

    // Generate each chapter
    for (const chapter of sortedChapters) {
      await this.generateChapter(chapter);
    }

    // Generate introduction and other foundational content
    await this.generateIntroduction();
    await this.generateTableOfContents();

    console.log('Finished generating new content.');
  }

  private async generateChapter(chapter: ChapterConfig): Promise<void> {
    const content = `---
title: ${chapter.title}
sidebar_position: ${chapter.position}
---

# ${chapter.title}

${chapter.content}

`;

    const filePath = path.join(this.config.targetDirectory, `${chapter.id}.md`);
    await fs.writeFile(filePath, content);
    console.log(`Generated chapter: ${filePath}`);
  }

  private async generateIntroduction(): Promise<void> {
    const introContent = `---
title: Introduction
sidebar_position: 1
---

# Introduction to ${this.config.title}

Welcome to the comprehensive guide on ${this.config.title}. This book is designed to provide you with a deep understanding of the subject matter, from fundamental concepts to advanced applications.

## About This Book

This book covers:

- Core principles and concepts
- Practical implementations
- Real-world applications
- Future directions and trends

## Who Should Read This Book

This book is intended for:

- Students and researchers in the field
- Practitioners and engineers
- Anyone interested in learning about this subject

Let's begin our journey into the fascinating world of ${this.config.title}.

`;

    const filePath = path.join(this.config.targetDirectory, 'introduction.md');
    await fs.writeFile(filePath, introContent);
    console.log(`Generated introduction: ${filePath}`);
  }

  private async generateTableOfContents(): Promise<void> {
    const tocContent = `---
title: Table of Contents
sidebar_position: 2
---

# Table of Contents

## Part I: Foundations
- [Introduction](./introduction.md)

## Part II: Core Concepts
${this.config.chapters
  .sort((a, b) => a.position - b.position)
  .map(chapter => `- [${chapter.title}](./${chapter.id}.md)`)
  .join('\n')}

## Part III: Advanced Topics
- Advanced Applications
- Future Directions
- Conclusion

`;

    const filePath = path.join(this.config.targetDirectory, 'toc.md');
    await fs.writeFile(filePath, tocContent);
    console.log(`Generated table of contents: ${filePath}`);
  }

  private async updateSidebar(): Promise<void> {
    console.log('Updating sidebar configuration...');

    // Read the current sidebar file - fix the path to be relative to docs directory
    const sidebarPath = path.join('docs', 'sidebars.ts');

    try {
      let sidebarContent = await fs.readFile(sidebarPath, 'utf-8');

      // Check if the category already exists to avoid duplicates
      if (sidebarContent.includes(`label: '${this.config.title}'`)) {
        console.log('Category already exists in sidebar, skipping update.');
        return;
      }

      // Update the sidebar to include the new content
      const newSidebarEntry = `    {
      type: 'category',
      label: '${this.config.title}',
      items: [
        '${this.config.targetDirectory.split('/').pop()}/introduction',
        '${this.config.targetDirectory.split('/').pop()}/toc',
        ${this.config.chapters
          .sort((a, b) => a.position - b.position)
          .map(chapter => `'${this.config.targetDirectory.split('/').pop()}/${chapter.id}'`)
          .join(',\n        ')}
      ],
    },`;

      // Find the closing bracket for tutorialSidebar array and insert before "Additional Specs" or at the end before the final ]
      const insertPosition = sidebarContent.lastIndexOf('    },') + 4; // After the last category closing
      const newContent =
        sidebarContent.substring(0, insertPosition) +
        ',\n' +
        newSidebarEntry +
        sidebarContent.substring(insertPosition);

      await fs.writeFile(sidebarPath, newContent);
      console.log('Updated sidebar configuration.');
    } catch (error) {
      console.error('Could not update sidebar configuration:', error);
    }
  }
}

// Example usage
async function runBookContentAgent(): Promise<void> {
  const config: BookContentConfig = {
    title: "Humanoid Agentic Systems",
    author: "AI Assistant",
    targetDirectory: "docs/docs/humanoid-book",
    chapters: [
      {
        id: "chapter-1-fundamentals",
        title: "Fundamentals of Humanoid Robotics",
        content: `## Introduction to Humanoid Robotics

Humanoid robotics represents one of the most ambitious fields in robotics engineering, aiming to create robots with human-like form and capabilities.

### Key Components

1. Mechanical Structure
2. Sensory Systems
3. Control Systems
4. Cognitive Capabilities

### Historical Context

The development of humanoid robots has evolved significantly since the early 20th century...`,
        position: 3
      },
      {
        id: "chapter-2-ai-integration",
        title: "AI Integration in Humanoid Systems",
        content: `## Artificial Intelligence in Humanoid Systems

Modern humanoid robots rely heavily on artificial intelligence to achieve human-like behavior and capabilities.

### Machine Learning Applications

- Perception and Recognition
- Decision Making
- Natural Language Processing
- Motor Control`,
        position: 4
      },
      {
        id: "chapter-3-control-systems",
        title: "Control Systems and Locomotion",
        content: `## Control Systems for Humanoid Robots

The challenge of controlling humanoid robots lies in managing their multiple degrees of freedom while maintaining balance and stability.

### Balance and Locomotion

Maintaining balance is one of the most critical challenges in humanoid robotics...`,
        position: 5
      },
      {
        id: "chapter-4-applications",
        title: "Applications and Future Directions",
        content: `## Applications of Humanoid Robotics

Humanoid robots have found applications in various domains:

- Healthcare assistance
- Customer service
- Educational tools
- Research platforms

### Future Directions

The future of humanoid robotics holds exciting possibilities...`,
        position: 6
      }
    ]
  };

  const agent = new BookContentAgent(config);
  await agent.generateBookContent();
}

// Only run if this file is executed directly
// In ES modules, we can check import.meta.url to determine if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runBookContentAgent().catch(console.error);
}

export { BookContentAgent, BookContentConfig, ChapterConfig };