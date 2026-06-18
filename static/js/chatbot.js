// Chatbot Functions
function toggleChatbot() {
  const chatWindow = document.getElementById('chatbot-window');
  const chatToggle = document.getElementById('chatbot-toggle');
  
  if (chatWindow.style.display === 'flex') {
    chatWindow.style.display = 'none';
    chatToggle.style.display = 'flex';
  } else {
    chatWindow.style.display = 'flex';
    chatToggle.style.display = 'none';
  }
}

function sendMessage(message) {
  const messagesContainer = document.getElementById('chatbot-messages');
  
  // Add user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chatbot-message user-message';
  userMsg.innerHTML = `
    <div class="message-content">
      <p>${message}</p>
    </div>
    <div class="message-avatar">You</div>
  `;
  messagesContainer.appendChild(userMsg);
  
  // Simulate bot response
  setTimeout(() => {
    const botMsg = document.createElement('div');
    botMsg.className = 'chatbot-message bot-message';
    let response = getBotResponse(message);
    botMsg.innerHTML = `
      <div class="message-avatar">Li</div>
      <div class="message-content">
        <p>${response}</p>
      </div>
    `;
    messagesContainer.appendChild(botMsg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }, 1000);
  
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function sendUserMessage() {
  const input = document.getElementById('chatbot-input-field');
  const message = input.value.trim();
  
  if (message) {
    sendMessage(message);
    input.value = '';
  }
}

function handleKeyPress(event) {
  if (event.key === 'Enter') {
    sendUserMessage();
  }
}

function getBotResponse(message) {
  const lowerMsg = message.toLowerCase();
  
  if (lowerMsg.includes('tnbc') || lowerMsg.includes('triple negative')) {
    return 'TNBC (Triple-Negative Breast Cancer) is a type of breast cancer that tests negative for estrogen receptors, progesterone receptors, and HER2. It represents about 15-20% of all breast cancers and requires specialized treatment approaches.';
  } else if (lowerMsg.includes('work') || lowerMsg.includes('classification')) {
    return 'Our system uses machine learning (XGBoost) to analyze your genomic data and classify your tumor into one of four TNBC molecular subtypes: BL1 (Basal-Like 1), BL2 (Basal-Like 2), LAR (Luminal Androgen Receptor), or M (Mesenchymal). We also provide SHAP explanations to help you understand which genetic features influenced the classification.';
  } else if (lowerMsg.includes('data') || lowerMsg.includes('need') || lowerMsg.includes('upload')) {
    return 'You can either upload a gene expression file (CSV/TSV format) or manually enter key genomic markers like BRCA1, TP53, and MYC expression levels. Clinical information such as age, stage, and tumor size also helps improve accuracy.';
  } else if (lowerMsg.includes('subtype')) {
    return 'We classify TNBC into four molecular subtypes: Basal-Like 1 (BL1), Basal-Like 2 (BL2), Luminal Androgen Receptor (LAR), and Mesenchymal (M). Each subtype responds differently to treatments.';
  } else if (lowerMsg.includes('trial') || lowerMsg.includes('clinical')) {
    return 'Based on your molecular subtype classification, we match you with relevant clinical trials. We consider factors like subtype compatibility, trial phase, location, and eligibility criteria to find the best matches for you.';
  } else if (lowerMsg.includes('shap') || lowerMsg.includes('explain')) {
    return 'SHAP (SHapley Additive exPlanations) values show which genetic features most influenced your classification. Higher SHAP values mean that feature pushed strongly toward your assigned subtype. This helps you understand the "why" behind your results.';
  } else if (lowerMsg.includes('start') || lowerMsg.includes('begin') || lowerMsg.includes('get started')) {
    return 'To get started: 1) Register/Login, 2) Go to "New Analysis", 3) Upload your genomic data or enter it manually, 4) Review your results and matched clinical trials. The whole process takes just a few minutes!';
  } else if (lowerMsg.includes('safe') || lowerMsg.includes('secure') || lowerMsg.includes('privacy')) {
    return 'Your data security is our priority. All data is encrypted end-to-end, stored securely, and complies with HIPAA and GDPR standards. We never share your personal or medical information without your explicit consent.';
  } else if (lowerMsg.includes('ar') || lowerMsg.includes('androgen receptor')) {
    return 'Androgen Receptor (AR) is an important IHC marker for TNBC subtyping. Strong, diffuse AR positivity (usually >10% staining) is the key clinical indicator of the Luminal Androgen Receptor (LAR) subtype. You can find AR status in your immunohistochemistry (IHC) report section of your pathology report.';
  } else if (lowerMsg.includes('egfr')) {
    return 'EGFR (Epidermal Growth Factor Receptor) is a basal-like marker. Positive EGFR expression helps identify basal-like TNBC subtypes (BL1/BL2). It\'s typically reported in your immunohistochemistry results along with other markers.';
  } else if (lowerMsg.includes('ck5') || lowerMsg.includes('cytokeratin') || lowerMsg.includes('basal marker')) {
    return 'Basal cytokeratins (CK5/6, CK14, CK17) are markers that help identify basal-like TNBC. Positive staining for these markers indicates a basal-like phenotype, which is clinically distinct from basal-marker-negative TNBC. These are found in your IHC report.';
  } else if (lowerMsg.includes('til') || lowerMsg.includes('lymphocyte') || lowerMsg.includes('immune')) {
    return 'TILs (Tumor-Infiltrating Lymphocytes) are immune cells within and around the tumor. High TILs (≥20%) are associated with better prognosis and higher response to immunotherapy, and are often seen in basal-like TNBC. TILs are assessed on H&E staining and reported as a percentage in your pathology report.';
  } else if (lowerMsg.includes('pd-l1') || lowerMsg.includes('pdl1')) {
    return 'PD-L1 is an immune checkpoint protein measured by IHC. Positive PD-L1 (≥1% staining) indicates potential benefit from immunotherapy checkpoint inhibitors and is more common in immune-rich basal-like tumours. It\'s reported in your IHC results if tested.';
  } else if (lowerMsg.includes('brca') || lowerMsg.includes('mutation')) {
    return 'BRCA1/BRCA2 mutations are inherited genetic changes that increase cancer risk. BRCA1 carriers are more likely to have basal-like TNBC. Your BRCA status comes from genetic testing (blood test), separate from the tumor pathology report. If you haven\'t been tested, select "Unknown/Not Tested".';
  } else if (lowerMsg.includes('histologic type') || lowerMsg.includes('ductal') || lowerMsg.includes('lobular')) {
    return 'Histologic type describes how the tumor cells look under the microscope. Most TNBC is "Invasive Ductal (NST)" but special types like apocrine often indicate LAR subtype, metaplastic suggests mesenchymal type, and medullary-like often has high immune infiltration. This is in your pathology report under "Diagnosis" or "Histologic Type".';
  } else if (lowerMsg.includes('lymphovascular') || lowerMsg.includes('lvi')) {
    return 'Lymphovascular invasion (LVI) means cancer cells have entered blood or lymph vessels. It\'s more common in aggressive basal-like tumors and affects prognosis. Your pathology report states whether LVI is "Present", "Absent", or "Cannot be determined".';
  } else if (lowerMsg.includes('menopausal')) {
    return 'Menopausal status helps predict subtype - LAR TNBC patients tend to be older/postmenopausal, while basal-like subtypes are more common in younger/premenopausal women. Your oncologist can tell you your menopausal status, or you can assess based on your age and menstrual history.';
  } else if (lowerMsg.includes('ki67') || lowerMsg.includes('ki-67')) {
    return 'Ki67 is a protein marker that indicates how fast cancer cells are growing. It\'s measured as a percentage and is usually found in your pathology report. A higher Ki67 (>30%) often indicates faster-growing tumors typical of basal-like (BL1/BL2) subtypes. Lower Ki67 is more common in LAR subtype. Ask your oncologist for this value if it\'s not in your report.';
  } else if (lowerMsg.includes('grade') || lowerMsg.includes('tumor grade')) {
    return 'Tumor grade (1-3) describes how abnormal the cancer cells look under a microscope. Grade 1 is well-differentiated (slower growing), Grade 2 is moderately differentiated, and Grade 3 is poorly differentiated (faster growing). You can find this in your pathology report under "Histologic Grade" or "Tumor Grade". Most TNBC is Grade 3, but low-grade variants may indicate LAR subtype.';
  } else if (lowerMsg.includes('pathology') || lowerMsg.includes('report')) {
    return 'Your pathology report contains all the clinical information needed: histologic type, tumor size, grade, stage, lymph node status, Ki67, and IHC markers like AR, EGFR, CK5/6, PD-L1, and TILs. You should receive this after your biopsy or surgery. If you don\'t have it, ask your oncologist or the hospital pathology department for a copy.';
  } else if (lowerMsg.includes('stage') || lowerMsg.includes('staging') || lowerMsg.includes('tnm')) {
    return 'Cancer stage (I-IV) indicates how advanced the cancer is based on tumor size (T), lymph node involvement (N), and metastasis (M). Stage I is early/localized, while Stage IV means it has spread to distant organs. Your oncologist determines this based on imaging and pathology. It\'s usually in your medical records or treatment plan.';
  } else if (lowerMsg.includes('lymph node')) {
    return 'Lymph node status tells if cancer has spread to nearby lymph nodes. N0 means negative (no cancer in nodes), N1 means 1-3 positive nodes, N2 means 4-9 nodes, and N3 means 10 or more positive nodes. This information comes from your pathology report after surgery or sentinel node biopsy.';
  } else if (lowerMsg.includes('where') || lowerMsg.includes('find')) {
    return 'All clinical information (histologic type, size, grade, stage, lymph nodes, Ki67, AR, EGFR, CK5/6, PD-L1, TILs) is in your pathology report and IHC results. If you\'re missing information, contact your oncologist\'s office or hospital medical records department. Some markers like BRCA status require separate genetic testing.';
  } else if (lowerMsg.includes('help') || lowerMsg.includes('hi') || lowerMsg.includes('hello')) {
    return "Hello! I'm here to help you understand TNBC classification, clinical markers, and how to fill out the analysis form. What would you like to know?";
  } else {
    return "I can help with questions about TNBC subtypes, clinical markers (Ki67, AR, EGFR, TILs, PD-L1), IHC markers, pathology reports, or how to find specific information in your medical records. What would you like to know?";
  }
}

// Note: button actions are wired via inline onclick handlers in
// chatbot_widget.html (toggleChatbot, sendMessage, sendUserMessage,
// handleKeyPress). Do NOT also addEventListener('click', toggleChatbot)
// here, as that would fire the toggle twice per click and cancel itself out.
