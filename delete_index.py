#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporary script: Delete llm_bench_rows index from Elasticsearch on port 9201
"""

import os
from elasticsearch import Elasticsearch

def delete_index():
    # Create ES client (port 9201)
    es = Elasticsearch(
        hosts=["http://localhost:9201"],
        timeout=30,
        max_retries=3,
        retry_on_timeout=True
    )
    
    index_name = "llm_bench_rows"
    
    try:
        # Test connection
        if not es.ping():
            print("❌ Cannot connect to Elasticsearch.")
            print("   - Check if port 9201 is running")
            print("   - Check if Elasticsearch service is started")
            return False
        
        print(f"✅ Connected to Elasticsearch on port 9201.")
        
        # Check if index exists
        if es.indices.exists(index=index_name):
            print(f"📋 Found index '{index_name}'.")
            
            # Display index information
            stats = es.indices.stats(index=index_name)
            doc_count = stats['indices'][index_name]['total']['docs']['count']
            size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
            size_mb = size_bytes / (1024 * 1024)
            
            print(f"   📊 Document count: {doc_count:,}")
            print(f"   💾 Size: {size_mb:.2f} MB")
            
            # User confirmation
            confirm = input(f"\n⚠️  Do you want to delete index '{index_name}'? (y/N): ")
            
            if confirm.lower() in ['y', 'yes']:
                # Delete index
                response = es.indices.delete(index=index_name)
                print(f"✅ Index '{index_name}' has been deleted.")
                print(f"   Response: {response}")
                return True
            else:
                print("❌ Index deletion cancelled.")
                return False
        else:
            print(f"ℹ️  Index '{index_name}' does not exist.")
            return True
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False

def list_indices():
    """Display all currently existing indices"""
    try:
        es = Elasticsearch(hosts=["http://localhost:9201"])
        
        if not es.ping():
            print("❌ Cannot connect to Elasticsearch.")
            return
        
        # Get all index information
        indices = es.indices.get_alias()
        
        if not indices:
            print("ℹ️  No indices currently exist.")
            return
        
        print("\n📋 Currently existing indices:")
        print("-" * 50)
        
        for index_name in indices.keys():
            try:
                stats = es.indices.stats(index=index_name)
                doc_count = stats['indices'][index_name]['total']['docs']['count']
                size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
                size_mb = size_bytes / (1024 * 1024)
                
                print(f"📁 {index_name}")
                print(f"   📊 Document count: {doc_count:,}")
                print(f"   💾 Size: {size_mb:.2f} MB")
                print()
            except:
                print(f"📁 {index_name} (Failed to retrieve info)")
                print()
                
    except Exception as e:
        print(f"❌ Error while listing indices: {e}")

if __name__ == "__main__":
    print("🔍 Delete llm_bench_rows index from Elasticsearch on port 9201")
    print("=" * 60)
    
    # Display current index list
    list_indices()
    
    print("\n" + "=" * 60)
    
    # Execute index deletion
    success = delete_index()
    
    if success:
        print("\n✅ Operation completed successfully.")
        
        # Display index list again after deletion
        print("\n📋 Index list after deletion:")
        list_indices()
    else:
        print("\n❌ Operation failed.")
    
    print("\n�� Exiting script.")
