/**
 * NETRA - Interactive Aerospace Landing Page Controller
 * Controls:
 * 1. Split Before/After Satellite Comparison Slider (10m Sentinel-2 vs 2.5m NETRA)
 * 2. Smooth Navigation and ScrollSpy for Sections
 * 3. Seamless transition between Landing Page and Geospatial Dashboard
 */

(function () {
  function initNetraLanding() {
    const landingRoot = document.getElementById('landing-page');
    const dashboardRoot = document.getElementById('dashboard-container');
    const sliderContainer = document.getElementById('netra-hero-slider');
    const lrImg = document.getElementById('netra-img-lr');
    const sliderHandle = document.getElementById('netra-slider-handle');
    const navBar = document.querySelector('.netra-navbar');

    // -------------------------------------------------------------
    // 1. BEFORE / AFTER COMPARISON SLIDER
    // -------------------------------------------------------------
    if (sliderContainer && lrImg && sliderHandle) {
      let isDragging = false;

      function updateSlider(xPos) {
        const rect = sliderContainer.getBoundingClientRect();
        let offsetX = xPos - rect.left;
        let percentage = (offsetX / rect.width) * 100;

        // Clamp between 1% and 99% for clean visual boundaries
        percentage = Math.max(1, Math.min(99, percentage));

        lrImg.style.clipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
        lrImg.style.webkitClipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
        sliderHandle.style.left = percentage + '%';
      }

      // Mouse drag handlers
      sliderContainer.addEventListener('mousedown', (e) => {
        isDragging = true;
        lrImg.style.transition = 'none';
        sliderHandle.style.transition = 'none';
        updateSlider(e.clientX);
      });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateSlider(e.clientX);
      });

      window.addEventListener('mouseup', () => {
        isDragging = false;
      });

      // Hover-follow mode (when not dragging, smoothly follow cursor over card)
      sliderContainer.addEventListener('mousemove', (e) => {
        if (isDragging) return;
        lrImg.style.transition = 'none';
        sliderHandle.style.transition = 'none';
        updateSlider(e.clientX);
      });

      // Smooth glide back to 50% when mouse leaves the card
      sliderContainer.addEventListener('mouseleave', () => {
        if (isDragging) return;
        lrImg.style.transition = 'clip-path 0.4s cubic-bezier(0.16, 1, 0.3, 1), -webkit-clip-path 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
        sliderHandle.style.transition = 'left 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
        lrImg.style.clipPath = 'polygon(0 0, 50% 0, 50% 100%, 0 100%)';
        lrImg.style.webkitClipPath = 'polygon(0 0, 50% 0, 50% 100%, 0 100%)';
        sliderHandle.style.left = '50%';
        setTimeout(() => {
          if (!isDragging) {
            lrImg.style.transition = 'none';
            sliderHandle.style.transition = 'none';
          }
        }, 400);
      });

      // Touch handlers for mobile/tablets
      sliderContainer.addEventListener('touchstart', (e) => {
        if (e.touches && e.touches[0]) {
          isDragging = true;
          lrImg.style.transition = 'none';
          sliderHandle.style.transition = 'none';
          updateSlider(e.touches[0].clientX);
        }
      }, { passive: true });

      window.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        if (e.touches && e.touches[0]) {
          updateSlider(e.touches[0].clientX);
        }
      }, { passive: true });

      window.addEventListener('touchend', () => {
        isDragging = false;
      });

      // Initial center position (50%)
      lrImg.style.clipPath = 'polygon(0 0, 50% 0, 50% 100%, 0 100%)';
      lrImg.style.webkitClipPath = 'polygon(0 0, 50% 0, 50% 100%, 0 100%)';
      sliderHandle.style.left = '50%';
    }

    // -------------------------------------------------------------
    // 2. LAUNCH DASHBOARD TRANSITION
    // -------------------------------------------------------------
    function openDashboard() {
      if (typeof window.netraLaunchDashboard === 'function') {
        window.netraLaunchDashboard();
        return;
      }
      if (!landingRoot || !dashboardRoot) return;

      landingRoot.classList.add('fade-out');
      setTimeout(() => {
        landingRoot.style.display = 'none';
        dashboardRoot.style.display = 'block';
        void dashboardRoot.offsetWidth; // Trigger reflow
        dashboardRoot.style.opacity = '1';

        // Trigger map and viewer resizes
        window.dispatchEvent(new Event('resize'));
        if (typeof window.netraResetViewers === 'function') {
          window.netraResetViewers();
        }
        if (typeof window.netraSelectPreset === 'function' && window.state && window.state.activePreset) {
          window.netraSelectPreset(window.state.activePreset);
        }
      }, 500);
    }

    // Bind all CTA buttons that open the demo / dashboard
    const openDemoBtns = [
      document.getElementById('btn-explore-demo'),
      document.getElementById('btn-nav-demo'),
      document.getElementById('btn-open-console'),
      document.getElementById('btn-start-analysis'),
      document.getElementById('btn-hero-launch-demo')
    ];

    openDemoBtns.forEach(btn => {
      if (btn) {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          openDashboard();
        });
      }
    });

    // Wire up preset launcher cards
    document.querySelectorAll('.netra-preset-card').forEach(card => {
      card.addEventListener('click', (e) => {
        e.preventDefault();
        const presetId = card.getAttribute('data-preset');
        openDashboard();
        setTimeout(() => {
          if (presetId === 'custom') {
            const btnModal = document.getElementById('open-map-modal') || document.getElementById('btn-open-map-modal') || document.getElementById('btn-custom-aoi');
            if (btnModal) btnModal.click();
          } else {
            if (typeof window.netraSelectPreset === 'function') {
              window.netraSelectPreset(presetId);
            } else {
              const chip = document.querySelector(`.aoi-card[data-id="${presetId}"]`);
              if (chip) chip.click();
            }
          }
        }, 550);
      });
    });

    // -------------------------------------------------------------
    // 3. RETURN TO LANDING PORTAL FROM DASHBOARD
    // -------------------------------------------------------------
    const backBtn = document.getElementById('btn-back-to-portal');
    if (backBtn) {
      backBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (typeof window.netraReturnToLanding === 'function') {
          window.netraReturnToLanding();
        } else if (landingRoot && dashboardRoot) {
          dashboardRoot.style.opacity = '0';
          setTimeout(() => {
            dashboardRoot.style.display = 'none';
            landingRoot.style.display = 'block';
            void landingRoot.offsetWidth;
            landingRoot.classList.remove('fade-out');
            landingRoot.scrollTop = 0;
          }, 300);
        }
      });
    }

    // -------------------------------------------------------------
    // 4. NAVBAR SCROLL EFFECT & SCROLLSPY
    // -------------------------------------------------------------
    if (landingRoot && navBar) {
      landingRoot.addEventListener('scroll', () => {
        if (landingRoot.scrollTop > 40) {
          navBar.classList.add('scrolled');
        } else {
          navBar.classList.remove('scrolled');
        }

        // Active link highlighting
        const sections = ['about', 'technology', 'use-cases', 'demo', 'team'];
        let currentSection = 'home';

        if (landingRoot.scrollTop < 300) {
          currentSection = 'home';
        } else {
          sections.forEach(id => {
            const sec = document.getElementById(id);
            if (sec) {
              const top = sec.offsetTop - 120;
              if (landingRoot.scrollTop >= top) {
                currentSection = id;
              }
            }
          });
        }

        document.querySelectorAll('.netra-nav-menu a').forEach(link => {
          const href = link.getAttribute('href');
          if (href === '#' && currentSection === 'home') {
            link.classList.add('active');
          } else if (href === '#' + currentSection) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      });
    }

    // Smooth scroll for nav anchor links inside landingRoot
    document.querySelectorAll('.netra-landing-root a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href').substring(1);
        if (!targetId) {
          e.preventDefault();
          landingRoot.scrollTo({ top: 0, behavior: 'smooth' });
          return;
        }

        const targetEl = document.getElementById(targetId);
        if (targetEl) {
          e.preventDefault();
          const targetOffset = targetEl.offsetTop - 70;
          landingRoot.scrollTo({ top: targetOffset, behavior: 'smooth' });
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNetraLanding);
  } else {
    initNetraLanding();
  }
})();
