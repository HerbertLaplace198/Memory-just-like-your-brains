const { Notice, Plugin, PluginSettingTab, Setting } = require("obsidian");
const { execFile } = require("child_process");
const path = require("path");

const DEFAULT_SETTINGS = {
  pythonPath: "/usr/bin/python3",
  scriptPath: "",
  memoryRoot: "",
};

module.exports = class NeuralMemoryReviewPlugin extends Plugin {
  async onload() {
    await this.loadSettings();
    this.running = false;
    this.addSettingTab(new NeuralMemoryReviewSettingTab(this.app, this));
    this.addRibbonIcon("check-circle", "Submit Neural Memory review decisions", () => {
      this.runSync(true);
    });
    this.addCommand({
      id: "sync-review-decisions",
      name: "Submit selected review decisions",
      callback: () => this.runSync(true),
    });
    this.registerMarkdownCodeBlockProcessor("neural-memory-submit", (_source, el) => {
      const button = el.createEl("button", { text: "Submit review decisions" });
      button.addEventListener("click", () => this.runSync(true));
    });
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    const vaultRoot = this.app.vault.adapter.basePath;
    if (!this.settings.memoryRoot) this.settings.memoryRoot = vaultRoot;
    if (!this.settings.scriptPath) {
      this.settings.scriptPath = path.resolve(vaultRoot, "../app/neural_memory.py");
    }
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  runSync(showNoChanges) {
    if (this.running) {
      new Notice("Neural Memory review submission is already running.");
      return;
    }
    this.running = true;
    execFile(
      this.settings.pythonPath,
      [this.settings.scriptPath, "--root", this.settings.memoryRoot, "sync-obsidian"],
      { timeout: 30000, maxBuffer: 1024 * 1024 },
      (error, stdout, stderr) => {
        this.running = false;
        if (error) {
          new Notice(`Neural Memory sync failed: ${stderr || error.message}`, 10000);
        } else {
          try {
            const result = JSON.parse(stdout);
            const reviews = result.memory_reviews || {};
            const changed =
              Number(reviews.confirmed || 0) +
              Number(reviews.needs_revision || 0) +
              Number(reviews.rejected || 0) +
              Number(reviews.concepts_confirmed || 0) +
              Number(reviews.concepts_rejected || 0) +
              Number(reviews.concepts_merged || 0) +
              Number(reviews.concepts_kept_distinct || 0) +
              Number(reviews.families_confirmed || 0) +
              Number(reviews.families_rejected || 0);
            const errors = reviews.errors || [];
            if (errors.length) {
              new Notice(`Neural Memory review error: ${errors.join("; ")}`, 10000);
            } else if (changed) {
              new Notice(`Neural Memory synchronized ${changed} review decision(s).`);
            } else if (showNoChanges) {
              new Notice("Neural Memory: no new review decisions.");
            }
          } catch (parseError) {
            new Notice(`Neural Memory returned invalid output: ${parseError.message}`, 10000);
          }
        }
      }
    );
  }
};

class NeuralMemoryReviewSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Neural Memory Review" });
    new Setting(containerEl)
      .setName("Python executable")
      .addText((text) => text
        .setValue(this.plugin.settings.pythonPath)
        .onChange(async (value) => {
          this.plugin.settings.pythonPath = value.trim();
          await this.plugin.saveSettings();
        }));
    new Setting(containerEl)
      .setName("Neural Memory script")
      .addText((text) => text
        .setValue(this.plugin.settings.scriptPath)
        .onChange(async (value) => {
          this.plugin.settings.scriptPath = value.trim();
          await this.plugin.saveSettings();
        }));
    new Setting(containerEl)
      .setName("Memory root")
      .addText((text) => text
        .setValue(this.plugin.settings.memoryRoot)
        .onChange(async (value) => {
          this.plugin.settings.memoryRoot = value.trim();
          await this.plugin.saveSettings();
        }));
  }
}
