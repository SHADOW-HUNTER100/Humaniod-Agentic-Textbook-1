#!/usr/bin/env node

import fs from 'fs/promises';
import { BookContentAgent } from './agents/book-content-agent.js';

async function runAgent() {
  try {
    // Read configuration
    const configData = await fs.readFile('./agents/book-config.json', 'utf-8');
    const config = JSON.parse(configData);

    // Create and run the agent
    const agent = new BookContentAgent(config);
    await agent.generateBookContent();

    console.log('Book content generation completed successfully!');
  } catch (error) {
    console.error('Error running book content agent:', error);
    process.exit(1);
  }
}

runAgent();