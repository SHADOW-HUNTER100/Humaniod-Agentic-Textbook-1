#!/usr/bin/env node

import fs from 'fs/promises';
import { SpecContentAgent } from './agents/spec-content-agent.js';

async function runSpecAgent() {
  try {
    // Read configuration
    const configData = await fs.readFile('./agents/spec-config.json', 'utf-8');
    const config = JSON.parse(configData);

    console.log(`Generating ${config.specs.length} spec(s)...`);

    // Generate each spec
    for (const spec of config.specs) {
      console.log(`\nGenerating spec: ${spec.title} (ID: ${spec.id})`);

      // Create individual config for this spec
      const specConfig = {
        id: spec.id,
        title: spec.title,
        description: spec.description,
        targetDirectory: spec.targetDirectory,
        content: {
          spec: "Detailed spec content...",
          plan: "Detailed plan content...",
          tasks: "Detailed tasks content..."
        }
      };

      // Create and run the agent for this spec
      const agent = new SpecContentAgent(specConfig);
      await agent.generateSpecContent();
    }

    console.log('\nAll specs generated successfully!');
  } catch (error) {
    console.error('Error running spec content agent:', error);
    process.exit(1);
  }
}

runSpecAgent();