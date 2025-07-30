#!/usr/bin/env python3
"""
Demo script showing how the type generation works.
"""

import sys
from pathlib import Path

def main():
    """Demonstrate type generation from schema."""
    print("🔧 LlamaFarm Configuration Type Generator Demo")
    print("=" * 50)

    # Check if schema exists
    schema_path = Path(__file__).parent / "schema.yaml"
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return 1

    print(f"📁 Schema file: {schema_path}")

    # Import and run the type generator
    try:
        from generate_types import TypeGenerator

        # Create generator
        generator = TypeGenerator(schema_path)

        # Generate types
        print("\n🔄 Generating types from schema...")
        types_content = generator.generate_types()

        # Show what was generated
        print("\n📝 Generated type definitions:")
        print("-" * 30)

        # Extract class names from the generated content
        lines = types_content.split('\n')
        class_names = []
        for line in lines:
            if line.strip().startswith('class ') and line.strip().endswith('(TypedDict):'):
                class_name = line.strip().split()[1]
                class_names.append(class_name)

        for i, class_name in enumerate(class_names, 1):
            print(f"{i:2d}. {class_name}")

        print(f"\n✅ Generated {len(class_names)} TypedDict classes")

        # Show the actual generated file
        output_path = Path(__file__).parent / "config_types.py"
        if output_path.exists():
            print(f"\n📄 Generated file: {output_path}")
            print(f"📏 File size: {output_path.stat().st_size} bytes")

            # Show first few lines
            with open(output_path, 'r') as f:
                lines = f.readlines()[:10]
                print("\n📖 First 10 lines:")
                for line in lines:
                    print(f"   {line.rstrip()}")

        print("\n🎉 Type generation completed successfully!")
        print("\n💡 To regenerate types after schema changes:")
        print("   python generate_types.py")

        return 0

    except Exception as e:
        print(f"❌ Error during type generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())