// MAT-HPO Documentation JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeMobileMenu();
    initializeCodeBlocks();
    initializeScrollToTop();
    updateActiveNavigation();
});

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Handle anchor links
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Mobile menu functionality
function initializeMobileMenu() {
    const mobileToggle = document.querySelector('.mobile-toggle');
    const sidebar = document.querySelector('.sidebar');
    const content = document.querySelector('.content');
    
    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
        
        // Close sidebar when clicking outside
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768 && 
                !sidebar.contains(e.target) && 
                !mobileToggle.contains(e.target)) {
                sidebar.classList.remove('active');
            }
        });
        
        // Close sidebar when window is resized to desktop
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                sidebar.classList.remove('active');
            }
        });
    }
}

// Code block enhancements
function initializeCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        // Add copy button
        const pre = block.parentElement;
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋 Copy';
        copyButton.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: var(--secondary-color);
            color: white;
            border: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s ease;
        `;
        
        pre.style.position = 'relative';
        pre.appendChild(copyButton);
        
        // Show copy button on hover
        pre.addEventListener('mouseenter', () => {
            copyButton.style.opacity = '1';
        });
        
        pre.addEventListener('mouseleave', () => {
            copyButton.style.opacity = '0';
        });
        
        // Copy functionality
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                copyButton.innerHTML = '✅ Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = '📋 Copy';
                }, 2000);
            });
        });
        
        // Syntax highlighting (basic)
        highlightSyntax(block);
    });
}

// Basic syntax highlighting
function highlightSyntax(codeBlock) {
    let code = codeBlock.innerHTML;
    
    // Python syntax highlighting
    if (codeBlock.classList.contains('language-python') || 
        codeBlock.parentElement.classList.contains('language-python')) {
        
        // Keywords
        code = code.replace(/\b(class|def|if|else|elif|for|while|try|except|finally|import|from|return|yield|with|as|pass|break|continue|and|or|not|in|is|None|True|False|self)\b/g, 
            '<span style="color: #0000ff; font-weight: bold;">$1</span>');
        
        // Strings
        code = code.replace(/(["'])((?:\\.|(?!\1)[^\\])*?)\1/g, 
            '<span style="color: #008000;">$1$2$1</span>');
        
        // Comments
        code = code.replace(/#.*$/gm, 
            '<span style="color: #808080; font-style: italic;">$&</span>');
        
        // Numbers
        code = code.replace(/\b\d+\.?\d*\b/g, 
            '<span style="color: #800080;">$&</span>');
    }
    
    // Bash syntax highlighting
    if (codeBlock.classList.contains('language-bash') || 
        codeBlock.parentElement.classList.contains('language-bash')) {
        
        // Commands
        code = code.replace(/\b(cd|ls|mkdir|pip|python|git|curl|wget|sudo|apt|yum|docker|conda)\b/g, 
            '<span style="color: #0000ff; font-weight: bold;">$1</span>');
        
        // Flags
        code = code.replace(/(\s|^)(-{1,2}[a-zA-Z0-9-]+)/g, 
            '$1<span style="color: #800080;">$2</span>');
        
        // Comments
        code = code.replace(/#.*$/gm, 
            '<span style="color: #808080; font-style: italic;">$&</span>');
    }
    
    codeBlock.innerHTML = code;
}

// Scroll to top functionality
function initializeScrollToTop() {
    const scrollButton = document.createElement('button');
    scrollButton.innerHTML = '↑';
    scrollButton.className = 'scroll-to-top';
    scrollButton.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        background: var(--secondary-color);
        color: white;
        border: none;
        border-radius: 50%;
        font-size: 1.2rem;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    `;
    
    document.body.appendChild(scrollButton);
    
    // Show/hide scroll button
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            scrollButton.style.opacity = '1';
            scrollButton.style.visibility = 'visible';
        } else {
            scrollButton.style.opacity = '0';
            scrollButton.style.visibility = 'hidden';
        }
    });
    
    // Scroll to top
    scrollButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// Update active navigation based on current page
function updateActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath || 
            (currentPath.includes(link.getAttribute('href')) && link.getAttribute('href') !== '/')) {
            link.classList.add('active');
        }
    });
}

// Search functionality (if search is implemented)
function initializeSearch() {
    const searchInput = document.querySelector('#search-input');
    const searchResults = document.querySelector('#search-results');
    
    if (searchInput && searchResults) {
        let searchData = [];
        
        // Load search data (this would be generated from all pages)
        fetch('/search-data.json')
            .then(response => response.json())
            .then(data => {
                searchData = data;
            })
            .catch(error => console.log('Search data not available'));
        
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase().trim();
            
            if (query.length < 2) {
                searchResults.innerHTML = '';
                searchResults.style.display = 'none';
                return;
            }
            
            const results = searchData.filter(item => 
                item.title.toLowerCase().includes(query) ||
                item.content.toLowerCase().includes(query)
            ).slice(0, 10);
            
            if (results.length > 0) {
                searchResults.innerHTML = results.map(result => `
                    <div class="search-result">
                        <a href="${result.url}">
                            <h4>${highlightMatch(result.title, query)}</h4>
                            <p>${highlightMatch(result.excerpt, query)}</p>
                        </a>
                    </div>
                `).join('');
                searchResults.style.display = 'block';
            } else {
                searchResults.innerHTML = '<div class="no-results">No results found</div>';
                searchResults.style.display = 'block';
            }
        });
        
        // Close search results when clicking outside
        document.addEventListener('click', function(e) {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.style.display = 'none';
            }
        });
    }
}

// Highlight search matches
function highlightMatch(text, query) {
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
}

// Table of contents generation
function generateTableOfContents() {
    const tocContainer = document.querySelector('#table-of-contents');
    if (!tocContainer) return;
    
    const headings = document.querySelectorAll('.main-content h2, .main-content h3, .main-content h4');
    if (headings.length === 0) return;
    
    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';
    
    headings.forEach((heading, index) => {
        // Add ID if not present
        if (!heading.id) {
            heading.id = `heading-${index}`;
        }
        
        const listItem = document.createElement('li');
        listItem.className = `toc-item toc-${heading.tagName.toLowerCase()}`;
        
        const link = document.createElement('a');
        link.href = `#${heading.id}`;
        link.textContent = heading.textContent;
        link.addEventListener('click', function(e) {
            e.preventDefault();
            heading.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        });
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });
    
    tocContainer.appendChild(tocList);
}

// Initialize additional features when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    generateTableOfContents();
    initializeSearch();
    
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}