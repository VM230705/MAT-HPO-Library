# MAT-HPO Library Documentation

This directory contains the complete HTML documentation website for the MAT-HPO Library, similar to the structure found in projects like [DACBench](https://automl.github.io/DACBench/main/).

## 📁 Directory Structure

```
docs/
├── index.html                      # Main landing page
├── quickstart.html                 # Quick start guide
├── css/
│   └── style.css                  # Main stylesheet
├── js/
│   └── main.js                    # JavaScript functionality
├── api/
│   └── base-environment.html      # API reference for BaseEnvironment
├── examples/
│   └── simple.html               # Simple usage example
├── images/                        # Images and assets (empty)
└── tutorials/                     # Additional tutorials (empty)
```

## 🚀 How to Use

### Option 1: Local File Server (Recommended)

For the best experience with all features working properly:

```bash
# Navigate to the docs directory
cd MAT_HPO_LIB/docs

# Start a simple HTTP server (Python 3)
python -m http.server 8000

# Or with Python 2
python -m SimpleHTTPServer 8000

# Or with Node.js (if you have it installed)
npx serve .
```

Then open your browser to: `http://localhost:8000`

### Option 2: Direct File Access

You can also open the files directly in your browser:

```bash
# Open the main page
open MAT_HPO_LIB/docs/index.html

# Or on Linux
xdg-open MAT_HPO_LIB/docs/index.html

# Or on Windows
start MAT_HPO_LIB/docs/index.html
```

> **Note:** Some features (like syntax highlighting and search) may not work properly when opening files directly due to browser security restrictions.

## 📚 Documentation Pages

### 🏠 Main Pages
- **[index.html](index.html)** - Library overview, features, and quick examples
- **[quickstart.html](quickstart.html)** - Complete tutorial from installation to first optimization

### 🔍 API Reference
- **[api/base-environment.html](api/base-environment.html)** - Complete BaseEnvironment class documentation

### 🎯 Examples & Tutorials
- **[examples/simple.html](examples/simple.html)** - Neural network optimization example with full code

## 🎨 Design Features

### Visual Design
- **Modern, clean interface** inspired by professional documentation sites
- **Responsive design** that works on desktop, tablet, and mobile
- **Dark sidebar navigation** with highlighted active sections
- **Professional typography** using system fonts for fast loading

### Interactive Features
- **Syntax highlighting** for Python, Bash, and JSON code blocks
- **Copy buttons** on all code blocks for easy copying
- **Smooth scrolling** navigation and anchor links
- **Mobile-responsive** hamburger menu for smaller screens
- **Table of contents** auto-generation for long pages

### User Experience
- **Breadcrumb navigation** to show current location
- **Consistent sidebar** across all pages for easy navigation
- **Search-ready structure** (can be extended with search functionality)
- **Print-friendly** CSS for documentation printing

## 🔧 Customization

### Adding New Pages

1. Create a new HTML file using the existing templates
2. Copy the sidebar navigation from any existing page
3. Update the breadcrumb navigation
4. Add the page to the sidebar menu in all pages

### Modifying Styles

Edit `css/style.css` to customize:
- Colors and themes (CSS variables at the top)
- Layout and spacing
- Typography and fonts
- Responsive breakpoints

### Adding Functionality

Edit `js/main.js` to add:
- Custom interactive features
- Search functionality
- Analytics tracking
- Additional syntax highlighting

## 📱 Mobile Support

The documentation is fully responsive and includes:
- **Collapsible sidebar** on mobile devices
- **Touch-friendly navigation** with proper spacing
- **Readable typography** on small screens
- **Fast loading** on mobile connections

## 🌐 Deployment Options

### GitHub Pages
```bash
# If your repo is on GitHub, you can enable GitHub Pages
# Go to Settings > Pages > Source: docs folder
# Your docs will be available at: https://username.github.io/MAT_HPO_LIB/
```

### Netlify
```bash
# Deploy the docs folder to Netlify for free hosting
# Just drag and drop the docs folder to netlify.com/drop
```

### Custom Server
```bash
# Copy the docs folder to your web server
scp -r docs/ user@yourserver.com:/var/www/html/mat-hpo-docs/
```

## 🔍 SEO Features

The documentation includes:
- **Meta descriptions** for each page
- **Open Graph tags** for social media sharing
- **Semantic HTML structure** for better indexing
- **Fast loading times** with optimized assets
- **Mobile-friendly** responsive design

## 🎯 Future Enhancements

Potential additions for the documentation:
- **Search functionality** with client-side indexing
- **Interactive code examples** with live execution
- **API documentation** for all classes and methods
- **Video tutorials** embedded in pages
- **Multi-language support** for internationalization
- **Dark/light theme toggle**
- **Version selector** for different library versions

## 🤝 Contributing

To contribute to the documentation:

1. **Edit existing pages** by modifying the HTML files
2. **Add new examples** in the `examples/` directory
3. **Update API docs** in the `api/` directory
4. **Improve styling** by editing `css/style.css`
5. **Add features** by modifying `js/main.js`

## 📄 Template Structure

Each HTML page follows this structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Meta tags and title -->
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <button class="mobile-toggle">☰</button>
    
    <div class="container">
        <!-- Sidebar navigation -->
        <nav class="sidebar">...</nav>
        
        <!-- Main content -->
        <main class="content">
            <div class="content-header">
                <!-- Breadcrumb navigation -->
            </div>
            
            <div class="main-content">
                <!-- Page content -->
            </div>
        </main>
    </div>
    
    <script src="js/main.js"></script>
</body>
</html>
```

## 🚀 Getting Started

1. **Start a local server** (see instructions above)
2. **Open** `http://localhost:8000` in your browser
3. **Navigate** through the documentation using the sidebar
4. **Try the examples** to understand MAT-HPO usage

---

**MAT-HPO Documentation** - Professional, responsive, and comprehensive! 🎉