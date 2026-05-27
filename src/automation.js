async function runAutomation() {
  const urlParams = new URLSearchParams(window.location.search);
  const autoMode = urlParams.get('auto');
  if (!autoMode) return;

  // Create automation status container
  const statusDiv = document.createElement('div');
  statusDiv.id = 'automation-status';
  statusDiv.style.cssText = 'position:fixed; top:10px; left:10px; background:rgba(10, 15, 20, 0.95); color:#00D4AA; padding:15px; font-family:"JetBrains Mono", monospace; font-size:12px; z-index:999999; border:2px solid #00D4AA; border-radius:6px; max-width:450px; box-shadow:0 0 20px rgba(0,212,170,0.4); line-height:1.5;';
  statusDiv.innerHTML = '<strong>[Automation]</strong> Initializing...';
  document.body.appendChild(statusDiv);

  // Helper to log status
  function logStatus(msg) {
    statusDiv.innerHTML = `<strong>[Automation]</strong> ${msg}`;
    console.log(`[Automation] ${msg}`);
  }

  try {
    // Load database files so we have the same objects
    const models = await fetch('./src/data/models.json').then(r => r.json());
    const gpus = await fetch('./src/data/gpus.json').then(r => r.json());
    logStatus('Data files loaded.');

    // Helper to wait
    const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

    // Small delay to ensure app modules are loaded and bound to window
    await wait(1000);

    if (autoMode === 'landing') {
      logStatus('Landing page loaded. Ready.');
      await wait(300);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    } 
    else if (autoMode === 'modeA') {
      logStatus('Navigating to Mode A (model → specs)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function (app.js not initialized yet?)');
      }
      window.showMode('model');
      await wait(800);

      logStatus('Searching and selecting "Llama 3.1 70B"...');
      const targetModel = models.find(m => m.display_name.includes('Llama 3.1 70B'));
      if (!targetModel) {
        throw new Error('Llama 3.1 70B not found in database!');
      }

      // Populate search input to look realistic
      const modelInput = document.getElementById('model-search');
      modelInput.value = targetModel.display_name;

      // Select the model
      window.selectModel(targetModel);
      await wait(800);

      logStatus('Clicking Calculate Requirements...');
      document.getElementById('calc-model-btn').click();
      await wait(1200);

      logStatus('Locating GPU Compatibility section...');
      const gpuSectionList = document.querySelector('.gpu-compat-list');
      if (gpuSectionList) {
        const rows = gpuSectionList.querySelectorAll('.gpu-row');
        const rowCount = rows.length;
        
        // Let's check layout properties of the list
        const style = window.getComputedStyle(gpuSectionList);
        const maxHeight = style.maxHeight;
        const overflowY = style.overflowY;
        
        logStatus(`GPU list loaded.<br>• Count: ${rowCount} rows (out of 143)<br>• max-height: ${maxHeight}<br>• overflow-y: ${overflowY}`);
        
        // Scroll to GPU section
        gpuSectionList.scrollIntoView({ behavior: 'instant', block: 'center' });
        await wait(500);
      } else {
        throw new Error('GPU compatibility list container (.gpu-compat-list) not found!');
      }
      
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    } 
    else if (autoMode === 'modeB_dropdown') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Clicking GPU input and typing "RTX 40"...');
      const gpuInput = document.getElementById('gpu-search');
      gpuInput.focus();
      gpuInput.value = 'RTX 40';
      
      // Trigger input event to open dropdown
      gpuInput.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(800);

      const dropdown = document.getElementById('gpu-dropdown');
      const isOpened = dropdown.classList.contains('open');
      const itemsCount = dropdown.querySelectorAll('.dropdown-item').length;
      
      // Check if dropdown is cut off or scrollable
      const dropdownStyle = window.getComputedStyle(dropdown);
      const dropMaxHeight = dropdownStyle.maxHeight;
      const dropOverflow = dropdownStyle.overflowY;
      
      logStatus(`GPU Dropdown loaded.<br>• Status: ${isOpened ? 'OPEN' : 'CLOSED'}<br>• Items shown: ${itemsCount}<br>• max-height: ${dropMaxHeight}<br>• overflow-y: ${dropOverflow}`);
      
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'modeB_scrolled_dropdown') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Clicking GPU input (leaving it empty)...');
      const gpuInput = document.getElementById('gpu-search');
      gpuInput.focus();
      
      // Trigger input event with empty query to open the full list
      gpuInput.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(800);

      const dropdown = document.getElementById('gpu-dropdown');
      const isOpened = dropdown.classList.contains('open');
      const itemsCount = dropdown.querySelectorAll('.dropdown-item').length;
      
      logStatus(`GPU Dropdown loaded with ${itemsCount} items. Scrolling to bottom...`);
      dropdown.scrollTop = dropdown.scrollHeight;
      await wait(1000);

      const scrolledItems = dropdown.querySelectorAll('.dropdown-item');
      const bottomItem = scrolledItems[scrolledItems.length - 1];
      const bottomLabel = bottomItem ? bottomItem.querySelector('.dropdown-label').textContent : 'none';
      const bottomSub = bottomItem ? bottomItem.querySelector('.dropdown-sub').textContent : 'none';

      logStatus(`Dropdown scrolled to bottom.<br>• Bottom GPU: ${bottomLabel} (${bottomSub})<br>• Total item count: ${itemsCount}`);
      
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'modeB_drawer_test') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Searching for "RTX" in GPU input...');
      const gpuInput = document.getElementById('gpu-search');
      gpuInput.focus();
      gpuInput.value = 'RTX';
      gpuInput.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(800);

      logStatus('Selecting first option in GPU dropdown...');
      const dropdown = document.getElementById('gpu-dropdown');
      const firstItem = dropdown.querySelector('.dropdown-item');
      if (firstItem) {
        firstItem.dispatchEvent(new Event('mousedown', { bubbles: true }));
      } else {
        throw new Error('No GPU dropdown option found for "RTX"!');
      }
      await wait(800);

      logStatus('Clicking Find Compatible Models...');
      const calcBtn = document.querySelector('#page-specs button.calc-btn');
      if (calcBtn) {
        calcBtn.click();
      } else {
        throw new Error('Find Compatible Models button not found!');
      }
      await wait(1200);

      logStatus('Opening first model card result...');
      const firstCard = document.querySelector('#specs-results-content .model-card');
      if (firstCard) {
        firstCard.click();
        await wait(1000);
        
        const drawer = document.getElementById('drawer');
        const drawerStyle = window.getComputedStyle(drawer);
        const isDrawerOpen = drawer.classList.contains('open');
        
        logStatus(`Drawer opened.<br>• Width: ${drawerStyle.width}<br>• Open: ${isDrawerOpen}<br>Scrolling drawer content...`);
        // Scroll the drawer down a bit to show scrollbar active track
        drawer.scrollTop = 150;
        await wait(500);
      } else {
        throw new Error('No model card results found in Mode B!');
      }

      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'modeB_drawer') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Selecting "NVIDIA RTX 4090"...');
      const targetGpu = gpus.find(g => g.display_name.includes('NVIDIA RTX 4090'));
      if (!targetGpu) {
        throw new Error('NVIDIA RTX 4090 not found in database!');
      }
      const gpuInput = document.getElementById('gpu-search');
      gpuInput.value = targetGpu.display_name;
      window.selectGPU(targetGpu);
      await wait(800);

      logStatus('Clicking Find Compatible Models...');
      const calcBtn = document.querySelector('#page-specs button.calc-btn');
      if (calcBtn) {
        calcBtn.click();
      } else {
        throw new Error('Find Compatible Models button not found!');
      }
      await wait(1200);

      logStatus('Opening first model card result...');
      const firstCard = document.querySelector('#specs-results-content .model-card');
      if (firstCard) {
        firstCard.click();
        await wait(1000);
        
        // Let's inspect the drawer layout
        const drawer = document.getElementById('drawer');
        const drawerStyle = window.getComputedStyle(drawer);
        const isDrawerOpen = drawer.classList.contains('open');
        
        logStatus(`Compatibility drawer loaded.<br>• Class: ${drawer.className}<br>• Visible width: ${drawerStyle.width}<br>• Right offset: ${drawerStyle.right}`);
      } else {
        throw new Error('No model card results found in Mode B!');
      }

      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'vram_step1') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Verifying "OR INPUT VRAM" button next to GPU label...');
      const btn = document.getElementById('gpu-mode-btn');
      if (!btn) throw new Error('gpu-mode-btn not found');
      
      const label = document.getElementById('gpu-label-text');
      if (!label) throw new Error('gpu-label-text not found');
      
      const btnStyle = window.getComputedStyle(btn);
      if (btnStyle.display === 'none' || btnStyle.visibility === 'hidden' || btnStyle.opacity === '0') {
        throw new Error('gpu-mode-btn is not visible');
      }
      
      if (btn.textContent.trim() !== 'or input VRAM') {
        throw new Error(`gpu-mode-btn text is "${btn.textContent.trim()}", expected "or input VRAM"`);
      }

      logStatus('Step 1 complete: "or input VRAM" button is visible and active.');
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'vram_step2') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Clicking "OR INPUT VRAM" button...');
      const btn = document.getElementById('gpu-mode-btn');
      if (!btn) throw new Error('gpu-mode-btn not found');
      btn.click();
      await wait(500);

      logStatus('Verifying custom VRAM input state...');
      const input = document.getElementById('gpu-search');
      if (input.type !== 'number') {
        throw new Error(`Input type is "${input.type}", expected "number"`);
      }
      const label = document.getElementById('gpu-label-text');
      if (label.textContent.trim() !== 'Custom VRAM (GB)') {
        throw new Error(`Label is "${label.textContent.trim()}", expected "Custom VRAM (GB)"`);
      }

      logStatus('Typing "16" in custom VRAM field...');
      input.value = '16';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(500);

      logStatus('Step 2 complete: Custom VRAM input toggled and 16GB typed.');
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'vram_step3') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Clicking "OR INPUT VRAM" button and entering "16" VRAM...');
      const btn = document.getElementById('gpu-mode-btn');
      btn.click();
      await wait(300);
      const input = document.getElementById('gpu-search');
      input.value = '16';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(500);

      logStatus('Clicking "Find Compatible Models"...');
      const calcBtn = document.querySelector('#page-specs button.calc-btn');
      if (!calcBtn) throw new Error('Calc button not found');
      calcBtn.click();
      await wait(1200);

      logStatus('Verifying compatibility results populated...');
      const cards = document.querySelectorAll('#specs-results-content .model-card');
      if (cards.length === 0) {
        throw new Error('No compatibility model cards found!');
      }

      const cardNames = Array.from(cards).map(c => {
        const nameEl = c.querySelector('.model-card-name') || c.querySelector('.model-name');
        return nameEl ? nameEl.textContent.trim() : '';
      }).filter(Boolean);

      logStatus(`Step 3 complete: Found ${cards.length} compatible models.<br>• Examples: ${cardNames.slice(0, 3).join(', ')}`);
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
    else if (autoMode === 'vram_step4') {
      logStatus('Navigating to Mode B (specs → model)...');
      if (typeof window.showMode !== 'function') {
        throw new Error('window.showMode is not a function');
      }
      window.showMode('specs');
      await wait(800);

      logStatus('Toggling VRAM input, typing "16", finding models...');
      const btn = document.getElementById('gpu-mode-btn');
      btn.click();
      await wait(300);
      const input = document.getElementById('gpu-search');
      input.value = '16';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await wait(500);

      const calcBtn = document.querySelector('#page-specs button.calc-btn');
      calcBtn.click();
      await wait(1200);

      logStatus('Clicking "or search GPUs" to return to GPU search mode...');
      btn.click();
      await wait(800);

      logStatus('Verifying inputs toggled back correctly...');
      if (input.type !== 'text') {
        throw new Error(`Input type is "${input.type}", expected "text"`);
      }
      const label = document.getElementById('gpu-label-text');
      if (label.textContent.trim() !== 'GPU') {
        throw new Error(`Label is "${label.textContent.trim()}", expected "GPU"`);
      }

      logStatus('Step 4 complete: Successfully toggled back to GPU search mode.');
      await wait(500);
      statusDiv.innerHTML += '<br><strong style="color:#00D4AA; text-shadow:0 0 5px #00D4AA">READY_FOR_SCREENSHOT</strong>';
    }
  } catch (error) {
    statusDiv.style.borderColor = '#EF4444';
    statusDiv.style.color = '#EF4444';
    statusDiv.innerHTML = `<strong>[Automation Error]</strong> ${error.message}<br><small>${error.stack}</small>`;
    console.error('[Automation Error]', error);
  }
}

// Run automation if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runAutomation);
} else {
  runAutomation();
}
