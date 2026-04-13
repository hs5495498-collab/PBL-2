/* ============================================================
   Smart Wardrobe Recommendation System — script.js
   Handles: selection toggles, file upload preview,
            recommendation generation with loading animation
   ============================================================ */

/* ── State ── */
let selectedOccasion = 'Casual';
let selectedWeather  = 'Hot';
let uploadedFiles    = [];

/* ============================================================
   FOOTER YEAR
   ============================================================ */
document.getElementById('year').textContent = new Date().getFullYear();

/* ============================================================
   OCCASION PILLS — toggle selection
   ============================================================ */
const occasionGroup = document.getElementById('occasionGroup');

occasionGroup.querySelectorAll('.pill').forEach(pill => {
  pill.addEventListener('click', () => {
    // Remove active from all, add to clicked
    occasionGroup.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    selectedOccasion = pill.dataset.value;
    console.log('Occasion:', selectedOccasion);
  });
});

/* ============================================================
   WEATHER CARDS — toggle selection
   ============================================================ */
const weatherGroup = document.getElementById('weatherGroup');

weatherGroup.querySelectorAll('.weather-card').forEach(card => {
  card.addEventListener('click', () => {
    weatherGroup.querySelectorAll('.weather-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    selectedWeather = card.dataset.value;
    console.log('Weather:', selectedWeather);
  });
});

/* ============================================================
   FILE UPLOAD — drag & drop + click to browse
   ============================================================ */
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const browseBtn  = document.getElementById('browseBtn');
const previewGrid = document.getElementById('previewGrid');

/* Click the browse link → trigger file input */
browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

/* Clicking anywhere on dropzone also opens file picker */
dropZone.addEventListener('click', () => fileInput.click());

/* File input change */
fileInput.addEventListener('change', (e) => {
  handleFiles(e.target.files);
});

/* Drag over styling */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

/* Drop event */
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

/**
 * Handle selected / dropped files:
 * - Filter for image types
 * - Generate preview thumbnails
 */
function handleFiles(fileList) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];

  Array.from(fileList).forEach(file => {
    if (!allowed.includes(file.type)) return; // skip non-images

    // Store file reference
    uploadedFiles.push(file);

    // Create preview element
    const reader = new FileReader();
    reader.onload = (e) => {
      const item = document.createElement('div');
      item.className = 'preview-item';

      const img = document.createElement('img');
      img.src = e.target.result;
      img.alt = file.name;

      const removeBtn = document.createElement('button');
      removeBtn.className = 'preview-remove';
      removeBtn.title = 'Remove';
      removeBtn.textContent = '✕';

      removeBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        // Remove from state array
        uploadedFiles = uploadedFiles.filter(f => f !== file);
        item.remove();
      });

      item.appendChild(img);
      item.appendChild(removeBtn);
      previewGrid.appendChild(item);
    };
    reader.readAsDataURL(file);
  });
}

/* ============================================================
   RECOMMENDATION ENGINE
   ============================================================ */

/* Outfit database keyed by [occasion][weather] */
const outfits = {
  Casual: {
    Hot:   { pieces: ['White linen tee', 'Light chino shorts', 'Slip-on sneakers', 'Sunglasses'],      note: 'Breathable fabrics keep you cool while looking effortlessly chic.' },
    Cold:  { pieces: ['Chunky knit sweater', 'Straight-leg jeans', 'Chelsea boots', 'Beige scarf'],   note: 'Layer up in warm tones for a cozy yet put-together look.' },
    Rainy: { pieces: ['Waterproof anorak', 'Dark slim jeans', 'White sneakers', 'Mini backpack'],      note: 'Stay dry without sacrificing style — keep it functional and fresh.' },
    Windy: { pieces: ['Trench coat', 'Black turtleneck', 'Straight jeans', 'Ankle boots'],            note: 'A classic trench is your best shield against gusty days.' },
    Cloudy:{ pieces: ['Oversized hoodie', 'High-waist jeans', 'Clean white sneakers', 'Cap'],         note: 'Relaxed silhouettes work beautifully on moody, overcast days.' },
  },
  Formal: {
    Hot:   { pieces: ['Light linen blazer', 'Crisp white shirt', 'Slim trousers', 'Loafers'],         note: 'Opt for breathable linen to stay sharp even in the heat.' },
    Cold:  { pieces: ['Wool suit', 'Silk tie', 'Oxford shoes', 'Cashmere pocket square'],             note: 'A well-tailored wool suit is the ultimate cold-weather power move.' },
    Rainy: { pieces: ['Dark suit', 'Trench coat', 'Leather brogues', 'Compact umbrella'],             note: 'Dark colours hide splashes; a tailored trench elevates the look.' },
    Windy: { pieces: ['Double-breasted blazer', 'Pressed dress shirt', 'Wool trousers', 'Oxfords'],   note: 'Double-breasted jackets stay closed and look intentional in wind.' },
    Cloudy:{ pieces: ['Charcoal suit', 'Light blue shirt', 'Tie', 'Derby shoes'],                    note: 'Charcoal reads authoritative on grey, cloudy backdrops.' },
  },
  Party: {
    Hot:   { pieces: ['Floral midi dress', 'Block-heel sandals', 'Mini clutch', 'Gold hoops'],        note: 'Bold prints and metallic accents are party-perfect in the heat.' },
    Cold:  { pieces: ['Velvet blazer', 'Black turtleneck', 'Slim trousers', 'Ankle boots'],           note: 'Velvet adds instant luxury — pair it with understated basics.' },
    Rainy: { pieces: ['Sequin mini skirt', 'Black bodysuit', 'Block-heel boots', 'Statement earrings'], note: 'Keep the drama on the outfit, not the weather.' },
    Windy: { pieces: ['Leather jacket', 'Satin slip dress', 'Heeled mules', 'Bold lip'],              note: 'Edgy meets elegant — the leather jacket is your wind warrior.' },
    Cloudy:{ pieces: ['Wrap midi dress', 'Wedge heels', 'Chain bag', 'Layered necklaces'],            note: 'Effortlessly chic: a wrap dress flatters every silhouette.' },
  },
  Sports: {
    Hot:   { pieces: ['Moisture-wicking tee', 'Running shorts', 'Trail runners', 'Sports cap'],       note: 'Go lightweight and breathable — performance first, style second.' },
    Cold:  { pieces: ['Thermal base layer', 'Fleece joggers', 'Running jacket', 'Beanie'],            note: 'Layer strategically: start warm, peel back as you heat up.' },
    Rainy: { pieces: ['Waterproof running jacket', 'Compression tights', 'Trail shoes', 'Headband'], note: 'Waterproofing is non-negotiable; bright colours improve visibility.' },
    Windy: { pieces: ['Wind-resistant jacket', 'Athletic leggings', 'Grip trainers', 'Buff'],         note: 'Tight-fitting pieces reduce drag on breezy outdoor days.' },
    Cloudy:{ pieces: ['Long-sleeve top', 'Sweat shorts', 'Cross-trainers', 'Wristband'],              note: 'Overcast skies = perfect workout weather. Keep it light and mobile.' },
  },
  Travel: {
    Hot:   { pieces: ['Breathable linen shirt', 'Travel shorts', 'Slip-on loafers', 'Straw hat'],     note: 'Pack light, look right — neutrals mix and match effortlessly.' },
    Cold:  { pieces: ['Puffer jacket', 'Merino wool base', 'Travel pants', 'Ankle boots'],            note: 'Merino wool regulates temperature and resists odour — travel gold.' },
    Rainy: { pieces: ['Packable rain jacket', 'Quick-dry trousers', 'Waterproof sneakers', 'Tote'],   note: 'Packable and practical: be ready for anything the sky brings.' },
    Windy: { pieces: ['Windbreaker', 'Straight jeans', 'Sturdy boots', 'Crossbody bag'],              note: 'Windbreakers fold into pockets — the perfect travel layer.' },
    Cloudy:{ pieces: ['Neutral hoodie', 'Comfortable jeans', 'White trainers', 'Backpack'],           note: 'Classics never let you down when exploring new places.' },
  },
  'Date Night': {
    Hot:   { pieces: ['Off-shoulder top', 'Linen trousers', 'Strappy sandals', 'Delicate necklace'], note: 'Effortless romance: keep it breezy and a little bare.' },
    Cold:  { pieces: ['Fitted turtleneck', 'Leather trousers', 'Block-heel boots', 'Mini bag'],       note: 'Monochrome tones in luxe fabrics spell instant sophistication.' },
    Rainy: { pieces: ['Silk blouse', 'Dark cigarette pants', 'Kitten heels', 'Trench coat'],          note: 'A silk blouse under a trench coat? Timeless romance unlocked.' },
    Windy: { pieces: ['Wrap dress', 'Leather jacket', 'Heeled booties', 'Simple ear cuffs'],          note: 'A wrap dress + leather jacket: the ultimate date night armour.' },
    Cloudy:{ pieces: ['Midi slip dress', 'Fitted blazer', 'Mule heels', 'Pearl studs'],               note: 'Pearl accents and a slip dress give quiet-luxury date night energy.' },
  },
};

/* Elements */
const generateBtn   = document.getElementById('generateBtn');
const resultCard    = document.getElementById('resultCard');
const resultLoading = document.getElementById('resultLoading');
const resultContent = document.getElementById('resultContent');
const outfitPieces  = document.getElementById('outfitPieces');
const resultNote    = document.getElementById('resultNote');
const retryBtn      = document.getElementById('retryBtn');

/**
 * Build and display the recommendation
 */
function generateRecommendation() {
  // 1. Show result card with loading state
  resultCard.classList.add('visible');
  resultLoading.style.display = 'flex';
  resultContent.classList.remove('show');

  // Smooth scroll to result
  setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50);

  // 2. Simulate AI processing delay (1.8s)
  setTimeout(() => {
    // Pick outfit based on state
    const outfit = (outfits[selectedOccasion] || outfits['Casual'])[selectedWeather] ||
                   outfits['Casual']['Hot'];

    // Populate pieces
    outfitPieces.innerHTML = '';
    outfit.pieces.forEach((piece, i) => {
      const tag = document.createElement('div');
      tag.className = 'piece-tag';
      tag.style.animationDelay = `${i * 0.08}s`;
      tag.textContent = '✦  ' + piece;
      outfitPieces.appendChild(tag);
    });

    // Note
    resultNote.textContent = outfit.note;

    // Hide loader, show content
    resultLoading.style.display = 'none';
    resultContent.classList.add('show');
  }, 1800);
}

/* CTA button click */
generateBtn.addEventListener('click', generateRecommendation);

/* Retry button — regenerate with current selections */
retryBtn.addEventListener('click', () => {
  resultContent.classList.remove('show');
  generateRecommendation();
});

/* ============================================================
   SUBTLE ENTRANCE ANIMATIONS FOR CARDS
   (IntersectionObserver — cards fade up as they scroll into view)
   ============================================================ */
const cards = document.querySelectorAll('.card');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

cards.forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(28px)';
  card.style.transition = `opacity 0.55s ${i * 0.1}s ease, transform 0.55s ${i * 0.1}s ease, box-shadow 0.28s ease`;
  observer.observe(card);
});
