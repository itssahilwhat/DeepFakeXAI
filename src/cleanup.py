import os
import ast
import logging
from pathlib import Path

class CodeAnalyzer:
    """Analyze codebase to find unused files and imports"""
    
    def __init__(self, src_dir="src"):
        self.src_dir = src_dir
        self.used_files = set()
        self.unused_files = set()
        
    def analyze_imports(self, file_path):
        """Extract all imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return imports
        except Exception as e:
            logging.warning(f"Could not parse {file_path}: {e}")
            return []
    
    def find_unused_files(self):
        """Find files that are not imported anywhere"""
        src_path = Path(self.src_dir)
        all_py_files = list(src_path.glob("*.py"))
        
        # Core files that are always used
        core_files = {
            "config.py", "model.py", "data.py", "train.py", "utils.py"
        }
        
        # Build dependency graph
        import_graph = {}
        for py_file in all_py_files:
            imports = self.analyze_imports(py_file)
            import_graph[py_file.name] = imports
        
        # Find which files are actually imported
        used_files = set()
        for file_name, imports in import_graph.items():
            for imp in imports:
                if imp.startswith('src.'):
                    module_name = imp.split('.')[-1] + '.py'
                    used_files.add(module_name)
                elif imp in [f.stem for f in all_py_files]:
                    used_files.add(imp + '.py')
        
        # Add core files
        used_files.update(core_files)
        
        # Find unused files
        all_file_names = {f.name for f in all_py_files}
        unused_files = all_file_names - used_files
        
        self.used_files = used_files
        self.unused_files = unused_files
        
        return unused_files
    
    def print_analysis(self):
        """Print analysis results"""
        print("\n" + "="*60)
        print("🔍 CODEBASE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📁 Analyzed directory: {self.src_dir}")
        print(f"✅ Used files ({len(self.used_files)}):")
        for file in sorted(self.used_files):
            print(f"   ✓ {file}")
        
        print(f"\n❌ Potentially unused files ({len(self.unused_files)}):")
        for file in sorted(self.unused_files):
            print(f"   ⚠ {file}")
        
        if self.unused_files:
            print(f"\n💡 Recommendations:")
            print("   - Review these files before deletion")
            print("   - Some may be used for future features")
            print("   - Check if they contain important utilities")
        else:
            print(f"\n🎉 All files appear to be used!")
        
        print("="*60)
    
    def get_unused_files_list(self):
        """Return list of unused files"""
        return sorted(self.unused_files)


def main():
    """Main function to run the analysis"""
    logging.basicConfig(level=logging.INFO)
    
    analyzer = CodeAnalyzer()
    unused_files = analyzer.find_unused_files()
    analyzer.print_analysis()
    
    # Ask user if they want to see details about specific files
    if unused_files:
        print(f"\nWould you like to see details about any specific unused file?")
        print("Enter filename (or 'all' for all, or 'none' to skip):")
        
        # For now, just show the analysis
        print("\nAnalysis complete. Review the results above.")
    
    return unused_files


if __name__ == "__main__":
    main() 