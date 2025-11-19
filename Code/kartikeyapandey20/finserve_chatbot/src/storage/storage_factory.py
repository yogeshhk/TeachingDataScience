"""
Storage Factory - Unified interface for FAISS or Milvus
Choose vector storage backend via configuration
"""
import os
from dotenv import load_dotenv
from typing import Literal, Optional

load_dotenv()


class VectorStorageFactory:
    """
    Factory for creating vector storage instances
    Supports: FAISS (local) or Milvus (cloud/local)
    """
    
    @staticmethod
    def create(
        storage_type: Literal['faiss', 'milvus'] = 'faiss',
        **kwargs
    ):
        """
        Create vector storage instance
        
        Args:
            storage_type: 'faiss' or 'milvus'
            **kwargs: Additional arguments for storage backend
            
        Returns:
            Storage instance (FAISSStorage or MilvusStorage)
            
        Examples:
            # Use FAISS (local, fast, no server)
            storage = VectorStorageFactory.create('faiss')
            
            # Use Milvus/Zilliz Cloud (reads from .env)
            storage = VectorStorageFactory.create('milvus')
            
            # Use Milvus with explicit credentials
            storage = VectorStorageFactory.create(
                'milvus',
                uri='https://...',
                token='...'
            )
        """
        
        if storage_type == 'faiss':
            from storage.faiss_storage import FAISSStorage
            return FAISSStorage(**kwargs)
        
        elif storage_type == 'milvus':
            from storage.milvus_storage import MilvusStorage
            
            # Default to Zilliz Cloud if credentials available
            use_cloud = bool(
                os.getenv('MILVUS_URI') or 
                os.getenv('ZILLIZ_CLOUD_URI') or
                kwargs.get('uri')
            )
            
            return MilvusStorage(
                use_zilliz_cloud=use_cloud,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unknown storage type: {storage_type}\n"
                f"Supported types: 'faiss', 'milvus'"
            )
    
    @staticmethod
    def get_default():
        """
        Get default storage based on environment
        
        Priority:
        1. If MILVUS_URI set in .env → Use Milvus
        2. Otherwise → Use FAISS
        """
        if os.getenv('MILVUS_URI') or os.getenv('ZILLIZ_CLOUD_URI'):
            print("✓ Milvus credentials found - using Milvus/Zilliz Cloud")
            return VectorStorageFactory.create('milvus')
        else:
            print("✓ No Milvus credentials - using FAISS (local)")
            return VectorStorageFactory.create('faiss')


# Configuration helper
def get_storage_config() -> dict:
    """Get current storage configuration from .env"""
    return {
        'milvus_uri': os.getenv('MILVUS_URI') or os.getenv('ZILLIZ_CLOUD_URI'),
        'milvus_token': os.getenv('MILVUS_TOKEN') or os.getenv('ZILLIZ_CLOUD_TOKEN'),
        'embedding_dim': int(os.getenv('EMBEDDING_DIM', '384')),
        'has_milvus': bool(os.getenv('MILVUS_URI') or os.getenv('ZILLIZ_CLOUD_URI'))
    }


def main():
    """Test storage factory"""
    
    print(f"{'='*70}")
    print("VECTOR STORAGE FACTORY")
    print(f"{'='*70}\n")
    
    # Check configuration
    config = get_storage_config()
    
    print("Current Configuration:")
    print(f"  Milvus URI: {'Set' if config['milvus_uri'] else 'Not set'}")
    print(f"  Milvus Token: {'Set' if config['milvus_token'] else 'Not set'}")
    print(f"  Embedding Dim: {config['embedding_dim']}")
    print(f"  Has Milvus: {config['has_milvus']}")
    
    print(f"\n{'='*70}")
    
    # Create default storage
    print("\nCreating default storage...")
    try:
        storage = VectorStorageFactory.get_default()
        print(f"✓ Storage created: {type(storage).__name__}")
        
        # Test operations
        stats = storage.get_stats()
        print(f"\nStorage stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo use Milvus/Zilliz Cloud:")
        print("1. Sign up at: https://cloud.zilliz.com/")
        print("2. Create a cluster")
        print("3. Add to .env:")
        print("   MILVUS_URI=https://your-cluster-uri")
        print("   MILVUS_TOKEN=your-token")
    
    print(f"\n{'='*70}")
    print("✅ Factory Test Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    main()
