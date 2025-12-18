# Copyright (C) 2025 Fabian Valle-simmons
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import json
import logging
import hashlib
import pickle
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import numpy as np

logger = logging.getLogger(__name__)


class InMemoryCache:
    """In-memory fallback cache when MongoDB is not available"""
    
    def __init__(self):
        self.general_reviews: Dict[str, Dict[str, Any]] = {}
        self.thread_analyses: Dict[str, Dict[str, Any]] = {}
        self.analysis_summaries: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, Dict[str, Any]] = {}
        self.embedding_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("💾 Using in-memory cache (MongoDB not available)")
    
    def _make_key(self, owner: str, repo: str, pr_number: int, commit_sha: str, 
                  indexing_paths: Optional[List[str]] = None) -> str:
        """Create a cache key"""
        key_parts = [owner, repo, str(pr_number), commit_sha]
        if indexing_paths:
            # Sort paths for consistent keys
            key_parts.append("|".join(sorted(indexing_paths)))
        return ":".join(key_parts)
    
    def get_general_review(self, owner: str, repo: str, pr_number: int, 
                          commit_sha: str, max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not commit_sha:
            logger.warning(f"⚠️ InMemoryCache.get_general_review: commit_sha is empty for {owner}/{repo}#{pr_number}")
            return None
        
        key = self._make_key(owner, repo, pr_number, commit_sha)
        logger.debug(f"🔍 InMemoryCache.get_general_review: key={key[:80]}..., total_keys={len(self.general_reviews)}")
        entry = self.general_reviews.get(key)
        if not entry:
            # Debug: show what keys we have
            matching_keys = [k for k in self.general_reviews.keys() if f"{owner}:{repo}:{pr_number}" in k]
            logger.debug(f"🔍 InMemoryCache: No exact match. Found {len(matching_keys)} keys for this PR: {[k[:80] for k in matching_keys[:3]]}")
            return None
        
        if max_age_hours:
            age_hours = (datetime.now(timezone.utc) - entry["created_at"]).total_seconds() / 3600
            if age_hours > max_age_hours:
                logger.debug(f"🔍 InMemoryCache: Entry is stale ({age_hours:.1f}h old)")
                return None
        
        logger.info(f"✅ Cache hit (in-memory): general review for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
        return entry.get("review_data")
    
    def store_general_review(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                            review_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not commit_sha:
            logger.warning(f"⚠️ Cannot cache review: commit_sha is empty for {owner}/{repo}#{pr_number}")
            return False
        
        if not review_data:
            logger.warning(f"⚠️ Cannot cache review: review_data is empty for {owner}/{repo}#{pr_number}")
            return False
        
        key = self._make_key(owner, repo, pr_number, commit_sha)
        logger.debug(f"💾 InMemoryCache.store_general_review: key={key[:80]}..., review_data keys={list(review_data.keys()) if isinstance(review_data, dict) else 'NOT_DICT'}")
        
        self.general_reviews[key] = {
            "review_data": review_data,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        logger.info(f"✅ Cached (in-memory): general review for {owner}/{repo}#{pr_number}@{commit_sha[:8]} (key: {key[:80]}..., total_cached: {len(self.general_reviews)})")
        return True
    
    def _clean_for_serialization(self, obj):
        """Recursively clean object for JSON/MongoDB serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_serialization(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert objects to dict
            return self._clean_for_serialization(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert to string as fallback
            return str(obj)
    
    def get_thread_analysis(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                           thread_id: str, max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        key = f"{self._make_key(owner, repo, pr_number, commit_sha)}:thread:{thread_id}"
        entry = self.thread_analyses.get(key)
        if not entry:
            return None
        
        if max_age_hours:
            age_hours = (datetime.now(timezone.utc) - entry["created_at"]).total_seconds() / 3600
            if age_hours > max_age_hours:
                return None
        
        return entry.get("analysis_data")
    
    def store_thread_analysis(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                             thread_id: str, analysis_data: Dict[str, Any]) -> bool:
        key = f"{self._make_key(owner, repo, pr_number, commit_sha)}:thread:{thread_id}"
        self.thread_analyses[key] = {
            "analysis_data": analysis_data,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        return True
    
    def get_analysis_summary(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                           max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        key = self._make_key(owner, repo, pr_number, commit_sha)
        entry = self.analysis_summaries.get(key)
        if not entry:
            return None
        
        if max_age_hours:
            age_hours = (datetime.now(timezone.utc) - entry["created_at"]).total_seconds() / 3600
            if age_hours > max_age_hours:
                return None
        
        return entry.get("summary_data")
    
    def store_analysis_summary(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                              summary_data: Dict[str, Any]) -> bool:
        key = self._make_key(owner, repo, pr_number, commit_sha)
        self.analysis_summaries[key] = {
            "summary_data": summary_data,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        return True
    
    def get_embeddings(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                      indexing_paths: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get cached embeddings (documents, embeddings array, metadata)"""
        key = self._make_key(owner, repo, pr_number, commit_sha, indexing_paths)
        entry = self.embeddings.get(key)
        if not entry:
            return None
        
        # Reconstruct numpy array from list
        embeddings_list = entry.get("embeddings")
        if embeddings_list:
            entry["embeddings"] = np.array(embeddings_list)
        
        logger.info(f"✅ Cache hit (in-memory): embeddings for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
        return entry
    
    def store_embeddings(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                        documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]],
                        indexing_paths: Optional[List[str]] = None) -> bool:
        """Store embeddings in cache"""
        key = self._make_key(owner, repo, pr_number, commit_sha, indexing_paths)
        
        # Convert numpy array to list for JSON serialization
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        self.embeddings[key] = {
            "documents": documents,
            "embeddings": embeddings_list,
            "metadata": metadata,
            "indexing_paths": indexing_paths or [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        logger.info(f"✅ Cached (in-memory): embeddings for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
        return True
    
    def get_embedding_metadata(self, owner: str, repo: str, pr_number: int, commit_sha: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(owner, repo, pr_number, commit_sha)
        return self.embedding_metadata.get(key)
    
    def store_embedding_metadata(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                                indexing_paths: Optional[List[str]] = None, total_files: int = 0,
                                total_chunks: int = 0) -> bool:
        key = self._make_key(owner, repo, pr_number, commit_sha)
        self.embedding_metadata[key] = {
            "indexing_paths": indexing_paths or [],
            "total_files": total_files,
            "total_chunks": total_chunks,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        return True
    
    def get_latest_cached_commit(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get the latest cached commit SHA for a PR"""
        # Find the most recent general review for this PR
        latest_entry = None
        latest_time = None
        
        for key, entry in self.general_reviews.items():
            # Parse key: owner:repo:pr_number:commit_sha[:indexing_paths]
            key_parts = key.split(":")
            if len(key_parts) >= 4:
                key_owner = key_parts[0]
                key_repo = key_parts[1]
                key_pr = key_parts[2]
                key_commit = key_parts[3]
                
                if key_owner == owner and key_repo == repo and key_pr == str(pr_number):
                    created_at = entry.get("created_at")
                    if created_at and (latest_time is None or created_at > latest_time):
                        latest_time = created_at
                        latest_entry = {
                            "commit_sha": key_commit,
                            "created_at": created_at,
                            "metadata": entry.get("metadata", {})
                        }
        
        return latest_entry
    
    def _clean_for_serialization(self, obj):
        """Recursively clean object for JSON/MongoDB serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_serialization(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, 'model_dump'):
            # Pydantic v2 model - use model_dump()
            return self._clean_for_serialization(obj.model_dump())
        elif hasattr(obj, 'dict'):
            # Pydantic v1 model - use dict()
            return self._clean_for_serialization(obj.dict())
        elif hasattr(obj, '__dict__'):
            # Convert other objects to dict
            return self._clean_for_serialization(obj.__dict__)
        else:
            # Convert to string as fallback
            return str(obj)
    
    def clear_commit_cache(self, owner: str, repo: str, pr_number: int, commit_sha: str) -> bool:
        """Clear all cached data for a specific commit"""
        prefix = f"{owner}:{repo}:{pr_number}:{commit_sha}"
        
        # Clear general reviews
        keys_to_remove = [k for k in self.general_reviews.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.general_reviews[key]
        
        # Clear thread analyses
        keys_to_remove = [k for k in self.thread_analyses.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.thread_analyses[key]
        
        # Clear analysis summaries
        keys_to_remove = [k for k in self.analysis_summaries.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.analysis_summaries[key]
        
        # Clear embeddings
        keys_to_remove = [k for k in self.embeddings.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.embeddings[key]
        
        # Clear embedding metadata
        keys_to_remove = [k for k in self.embedding_metadata.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.embedding_metadata[key]
        
        logger.info(f"🗑️ Cleared in-memory cache for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
        return True


class MongoCache:
    """MongoDB-based cache for AI review results with commit-level freshness tracking"""
    
    def _clean_for_serialization(self, obj):
        """Recursively clean object for JSON/MongoDB serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_serialization(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, 'model_dump'):
            # Pydantic v2 model - use model_dump()
            return self._clean_for_serialization(obj.model_dump())
        elif hasattr(obj, 'dict'):
            # Pydantic v1 model - use dict()
            return self._clean_for_serialization(obj.dict())
        elif hasattr(obj, '__dict__'):
            # Convert other objects to dict
            return self._clean_for_serialization(obj.__dict__)
        else:
            # Convert to string as fallback
            return str(obj)
    
    def __init__(self, mongo_uri: Optional[str] = None):
        """
        Initialize MongoDB cache.
        
        Args:
            mongo_uri: MongoDB connection URI. If None, reads from MONGODB_URI env var.
                     Falls back to mongodb://localhost:27017 if not set.
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.get_database("ai_code_review_cache")
            
            # Create indexes for faster lookups
            self.db.general_reviews.create_index([
                ("owner", 1), ("repo", 1), ("pr_number", 1), ("commit_sha", 1)
            ], unique=True)
            self.db.thread_analyses.create_index([
                ("owner", 1), ("repo", 1), ("pr_number", 1), ("commit_sha", 1), ("thread_id", 1)
            ], unique=True)
            self.db.analysis_summaries.create_index([
                ("owner", 1), ("repo", 1), ("pr_number", 1), ("commit_sha", 1)
            ], unique=True)
            
            # Index for freshness queries
            self.db.general_reviews.create_index("created_at")
            self.db.thread_analyses.create_index("created_at")
            self.db.analysis_summaries.create_index("created_at")
            
            # Create embedding_metadata collection and index if it doesn't exist
            if "embedding_metadata" not in self.db.list_collection_names():
                self.db.create_collection("embedding_metadata")
            self.db.embedding_metadata.create_index([
                ("owner", 1), ("repo", 1), ("pr_number", 1), ("commit_sha", 1)
            ], unique=True)
            self.db.embedding_metadata.create_index("created_at")
            
            # Create embeddings collection with compound index including indexing_paths
            if "embeddings" not in self.db.list_collection_names():
                self.db.create_collection("embeddings")
            self.db.embeddings.create_index([
                ("owner", 1), ("repo", 1), ("pr_number", 1), ("commit_sha", 1), ("indexing_paths_hash", 1)
            ], unique=True)
            self.db.embeddings.create_index("created_at")
            
            logger.info("✅ Connected to MongoDB cache")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"⚠️ Could not connect to MongoDB: {e}. Cache will be disabled.")
            self.client = None
            self.db = None
        except Exception as e:
            logger.error(f"❌ Error initializing MongoDB cache: {e}")
            self.client = None
            self.db = None
    
    def _is_connected(self) -> bool:
        """Check if MongoDB connection is active"""
        if self.client is None or self.db is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False
    
    def _hash_indexing_paths(self, indexing_paths: Optional[List[str]] = None) -> str:
        """Create a hash of indexing paths for consistent lookup"""
        if not indexing_paths:
            return "none"
        # Sort paths for consistent hashing
        sorted_paths = sorted(indexing_paths)
        return hashlib.md5("|".join(sorted_paths).encode()).hexdigest()
    
    def get_general_review(
        self, 
        owner: str, 
        repo: str, 
        pr_number: int, 
        commit_sha: str,
        max_age_hours: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached general review for a specific commit.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            commit_sha: Commit SHA (headRefOid)
            max_age_hours: Maximum age in hours. If None, returns any cached result.
        
        Returns:
            Cached review data or None if not found or stale
        """
        if not commit_sha:
            logger.warning(f"⚠️ MongoCache.get_general_review: commit_sha is empty for {owner}/{repo}#{pr_number}")
            return None
        
        if not self._is_connected():
            logger.debug(f"🔍 MongoCache.get_general_review: Not connected, returning None")
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            }
            
            logger.debug(f"🔍 MongoCache.get_general_review: query={query}")
            result = self.db.general_reviews.find_one(query)
            
            if not result:
                # Debug: check if there are any reviews for this PR
                any_review = self.db.general_reviews.find_one({
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number
                }, {"commit_sha": 1})
                if any_review:
                    logger.debug(f"🔍 MongoCache: No exact match. Found review for different commit: {any_review.get('commit_sha', 'unknown')[:8]}")
                else:
                    logger.debug(f"🔍 MongoCache: No reviews found for {owner}/{repo}#{pr_number}")
                return None
            
            # Check freshness if max_age_hours is specified
            if max_age_hours is not None:
                created_at = result.get("created_at")
                if created_at:
                    age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        logger.info(f"Cache entry for {owner}/{repo}#{pr_number}@{commit_sha[:8]} is stale ({age_hours:.1f}h old)")
                        return None
            
            # Remove MongoDB _id before returning
            result.pop("_id", None)
            review_data = result.get("review_data")
            logger.info(f"✅ Cache hit (MongoDB): general review for {owner}/{repo}#{pr_number}@{commit_sha[:8]}, review_data keys: {list(review_data.keys()) if isinstance(review_data, dict) else 'NOT_DICT'}")
            return review_data
        except Exception as e:
            logger.error(f"❌ Error reading from cache: {e}")
            return None
    
    def store_general_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        review_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store general review in cache.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            commit_sha: Commit SHA (headRefOid)
            review_data: The review result data
            metadata: Optional metadata (e.g., indexing_paths, custom_prompt)
        
        Returns:
            True if stored successfully, False otherwise
        """
        print(f"🔍 MongoCache.store_general_review: Checking preconditions...")
        print(f"🔍 MongoCache: commit_sha={commit_sha[:8] if commit_sha else 'EMPTY'}, review_data={bool(review_data)}, connected={self._is_connected()}")
        
        if not commit_sha:
            logger.warning(f"⚠️ MongoCache.store_general_review: commit_sha is empty for {owner}/{repo}#{pr_number}")
            print(f"❌ MongoCache: commit_sha is EMPTY!")
            return False
        
        if not review_data:
            logger.warning(f"⚠️ MongoCache.store_general_review: review_data is empty for {owner}/{repo}#{pr_number}")
            print(f"❌ MongoCache: review_data is EMPTY!")
            return False
        
        if not self._is_connected():
            logger.warning(f"⚠️ MongoDB not connected, cannot cache review for {owner}/{repo}#{pr_number}")
            print(f"❌ MongoCache: MongoDB NOT CONNECTED!")
            return False
        
        print(f"✅ MongoCache: All preconditions passed, proceeding with storage...")
        
        try:
            print(f"💾 MongoCache.store_general_review START: {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            logger.info(f"💾 MongoCache.store_general_review START: {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            
            # Always clean review_data to ensure MongoDB compatibility (BSON serialization)
            # This converts Pydantic models and other objects to plain dicts
            print(f"💾 MongoCache: Cleaning review_data for MongoDB serialization...")
            review_data = self._clean_for_serialization(review_data)
            print(f"💾 MongoCache: review_data cleaned successfully")
            
            # Also clean metadata if provided
            if metadata:
                metadata = self._clean_for_serialization(metadata)
            
            document = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "review_data": review_data,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            print(f"💾 MongoCache: Document prepared with keys: {list(document.keys())}")
            logger.info(f"💾 MongoCache.store_general_review: Storing document with keys: {list(document.keys())}, review_data keys: {list(review_data.keys()) if isinstance(review_data, dict) else 'NOT_DICT'}")
            
            # Use upsert to update if exists
            print(f"💾 MongoCache: Attempting MongoDB upsert...")
            result = self.db.general_reviews.update_one(
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha
                },
                {"$set": document},
                upsert=True
            )
            
            print(f"💾 MongoCache: Upsert result - inserted: {result.upserted_id is not None}, modified: {result.modified_count}, matched: {result.matched_count}")
            
            was_inserted = result.upserted_id is not None
            was_updated = result.modified_count > 0
            logger.info(f"✅ Cached (MongoDB): general review for {owner}/{repo}#{pr_number}@{commit_sha[:8]} (inserted: {was_inserted}, updated: {was_updated}, matched: {result.matched_count})")
            print(f"✅ MongoCache: Upsert successful - inserted={was_inserted}, updated={was_updated}")
            
            # Verify it was actually stored
            print(f"💾 MongoCache: Verifying storage...")
            verify = self.db.general_reviews.find_one({
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            })
            if verify:
                logger.info(f"✅ Verified cache storage successful for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
                print(f"✅ MongoCache: Verification successful!")
                return True
            else:
                logger.error(f"❌ Cache verification FAILED - document not found after insert!")
                print(f"❌ MongoCache: Verification FAILED - document not found!")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error storing in cache: {e}", exc_info=True)
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Full traceback: {error_trace}")
            print(f"❌ MongoCache EXCEPTION: {e}")
            print(f"❌ MongoCache TRACEBACK: {error_trace}")
            return False
    
    def get_thread_analysis(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        thread_id: str,
        max_age_hours: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached thread analysis for a specific commit"""
        if not self._is_connected():
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "thread_id": thread_id
            }
            
            result = self.db.thread_analyses.find_one(query)
            
            if not result:
                return None
            
            # Check freshness
            if max_age_hours is not None:
                created_at = result.get("created_at")
                if created_at:
                    age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        return None
            
            result.pop("_id", None)
            return result.get("analysis_data")
        except Exception as e:
            logger.error(f"❌ Error reading thread analysis from cache: {e}")
            return None
    
    def store_thread_analysis(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        thread_id: str,
        analysis_data: Dict[str, Any]
    ) -> bool:
        """Store thread analysis in cache"""
        if not self._is_connected():
            return False
        
        try:
            document = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "thread_id": thread_id,
                "analysis_data": analysis_data,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.db.thread_analyses.update_one(
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha,
                    "thread_id": thread_id
                },
                {"$set": document},
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.error(f"❌ Error storing thread analysis in cache: {e}")
            return False
    
    def get_analysis_summary(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        max_age_hours: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis summary for a specific commit"""
        if not self._is_connected():
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            }
            
            result = self.db.analysis_summaries.find_one(query)
            
            if not result:
                return None
            
            # Check freshness
            if max_age_hours is not None:
                created_at = result.get("created_at")
                if created_at:
                    age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        return None
            
            result.pop("_id", None)
            return result.get("summary_data")
        except Exception as e:
            logger.error(f"❌ Error reading analysis summary from cache: {e}")
            return None
    
    def store_analysis_summary(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        summary_data: Dict[str, Any]
    ) -> bool:
        """Store analysis summary in cache"""
        if not self._is_connected():
            return False
        
        try:
            document = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "summary_data": summary_data,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.db.analysis_summaries.update_one(
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha
                },
                {"$set": document},
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.error(f"❌ Error storing analysis summary in cache: {e}")
            return False
    
    def get_embeddings(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        indexing_paths: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached embeddings (documents, embeddings array, metadata).
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            commit_sha: Commit SHA
            indexing_paths: Optional list of paths that were indexed (for cache key)
        
        Returns:
            Dict with 'documents', 'embeddings' (numpy array), 'metadata', or None if not found
        """
        if not self._is_connected():
            return None
        
        try:
            indexing_paths_hash = self._hash_indexing_paths(indexing_paths)
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "indexing_paths_hash": indexing_paths_hash
            }
            
            result = self.db.embeddings.find_one(query)
            
            if not result:
                return None
            
            # Reconstruct numpy array from stored list
            embeddings_list = result.get("embeddings")
            if embeddings_list:
                result["embeddings"] = np.array(embeddings_list)
            
            result.pop("_id", None)
            logger.info(f"✅ Cache hit: embeddings for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            return result
        except Exception as e:
            logger.error(f"❌ Error reading embeddings from cache: {e}")
            return None
    
    def store_embeddings(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        indexing_paths: Optional[List[str]] = None
    ) -> bool:
        """
        Store embeddings in cache.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number
            commit_sha: Commit SHA
            documents: List of document texts
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts
            indexing_paths: Optional list of paths that were indexed
        
        Returns:
            True if stored successfully, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            indexing_paths_hash = self._hash_indexing_paths(indexing_paths)
            
            # Convert numpy array to list for MongoDB storage
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            document = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "indexing_paths_hash": indexing_paths_hash,
                "indexing_paths": indexing_paths or [],
                "documents": documents,
                "embeddings": embeddings_list,
                "metadata": metadata,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.db.embeddings.update_one(
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha,
                    "indexing_paths_hash": indexing_paths_hash
                },
                {"$set": document},
                upsert=True
            )
            
            logger.info(f"✅ Cached embeddings for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            return True
        except Exception as e:
            logger.error(f"❌ Error storing embeddings in cache: {e}")
            return False
    
    def get_commit_freshness(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str
    ) -> Optional[datetime]:
        """
        Get the freshness timestamp for a commit (earliest cached data for this commit).
        
        Returns:
            datetime of oldest cached data for this commit, or None if not cached
        """
        if not self._is_connected():
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            }
            
            # Check all collections for this commit
            timestamps = []
            
            general = self.db.general_reviews.find_one(query, {"created_at": 1})
            if general and general.get("created_at"):
                timestamps.append(general["created_at"])
            
            summary = self.db.analysis_summaries.find_one(query, {"created_at": 1})
            if summary and summary.get("created_at"):
                timestamps.append(summary["created_at"])
            
            threads = self.db.thread_analyses.find(query, {"created_at": 1})
            for thread in threads:
                if thread.get("created_at"):
                    timestamps.append(thread["created_at"])
            
            if timestamps:
                return min(timestamps)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting commit freshness: {e}")
            return None
    
    def clear_commit_cache(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str
    ) -> bool:
        """Clear all cached data for a specific commit"""
        if not self._is_connected():
            return False
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            }
            
            self.db.general_reviews.delete_many(query)
            self.db.thread_analyses.delete_many(query)
            self.db.analysis_summaries.delete_many(query)
            self.db.embeddings.delete_many(query)
            self.db.embedding_metadata.delete_many(query)
            
            logger.info(f"🗑️ Cleared cache for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            return False
    
    def get_latest_cached_commit(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest cached commit SHA for a PR (any commit).
        Returns the most recent cached commit with its metadata.
        
        Returns:
            Dict with 'commit_sha' and 'created_at', or None if no cache exists
        """
        if not self._is_connected():
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number
            }
            
            # Find the most recent general review for this PR
            latest_review = self.db.general_reviews.find_one(
                query,
                sort=[("created_at", -1)],
                projection={"commit_sha": 1, "created_at": 1, "metadata": 1}
            )
            
            if latest_review:
                return {
                    "commit_sha": latest_review.get("commit_sha"),
                    "created_at": latest_review.get("created_at"),
                    "metadata": latest_review.get("metadata", {})
                }
            
            return None
        except Exception as e:
            logger.error(f"❌ Error getting latest cached commit: {e}")
            return None
    
    def store_embedding_metadata(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        indexing_paths: Optional[List[str]] = None,
        total_files: int = 0,
        total_chunks: int = 0
    ) -> bool:
        """
        Store embedding metadata for a PR commit.
        This helps track what was indexed and when, so we can intelligently re-index if needed.
        """
        if not self._is_connected():
            return False
        
        try:
            document = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha,
                "indexing_paths": indexing_paths or [],
                "total_files": total_files,
                "total_chunks": total_chunks,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Create collection for embedding metadata if it doesn't exist
            if "embedding_metadata" not in self.db.list_collection_names():
                self.db.create_collection("embedding_metadata")
            
            self.db.embedding_metadata.update_one(
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha
                },
                {"$set": document},
                upsert=True
            )
            
            logger.info(f"✅ Stored embedding metadata for {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            return True
        except Exception as e:
            logger.error(f"❌ Error storing embedding metadata: {e}")
            return False
    
    def get_embedding_metadata(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str
    ) -> Optional[Dict[str, Any]]:
        """Get embedding metadata for a specific commit"""
        if not self._is_connected():
            return None
        
        try:
            query = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "commit_sha": commit_sha
            }
            
            result = self.db.embedding_metadata.find_one(query)
            if result:
                result.pop("_id", None)
                return result
            return None
        except Exception as e:
            logger.error(f"❌ Error getting embedding metadata: {e}")
            return None


class IntelligentCache:
    """
    Intelligent cache wrapper that uses MongoDB if available, falls back to in-memory cache.
    Provides a unified interface for caching embeddings, general reviews, and metadata.
    """
    
    def __init__(self, mongo_uri: Optional[str] = None):
        """Initialize cache with MongoDB if available, otherwise use in-memory"""
        self.mongo_cache = MongoCache(mongo_uri)
        self.memory_cache = InMemoryCache()
        
        # Use MongoDB if connected, otherwise use in-memory
        self._use_mongo = self.mongo_cache._is_connected()
        
        if self._use_mongo:
            logger.info("🚀 Using MongoDB cache (persistent)")
        else:
            logger.info("💾 Using in-memory cache (MongoDB not available)")
    
    def _get_cache(self):
        """Get the active cache backend"""
        # Re-check connection status in case it changed
        if self.mongo_cache._is_connected():
            if not self._use_mongo:
                logger.info("🔄 MongoDB connection restored, switching to MongoDB cache")
                self._use_mongo = True
            return self.mongo_cache
        else:
            if self._use_mongo:
                logger.warning("⚠️ MongoDB connection lost, falling back to in-memory cache")
                self._use_mongo = False
            return self.memory_cache
    
    # General Review methods
    def get_general_review(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                          max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached general review"""
        active_cache = self._get_cache()
        logger.debug(f"🔍 IntelligentCache.get_general_review: {owner}/{repo}#{pr_number}@{commit_sha[:8] if commit_sha else 'NONE'} (backend: {'MongoDB' if self._use_mongo else 'In-memory'})")
        result = active_cache.get_general_review(owner, repo, pr_number, commit_sha, max_age_hours)
        logger.debug(f"🔍 IntelligentCache.get_general_review result: {'FOUND' if result else 'NOT FOUND'}")
        return result
    
    def store_general_review(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                            review_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store general review in cache"""
        if not commit_sha:
            logger.warning(f"⚠️ IntelligentCache.store_general_review: commit_sha is empty for {owner}/{repo}#{pr_number}")
            return False
        
        if not review_data:
            logger.warning(f"⚠️ IntelligentCache.store_general_review: review_data is empty for {owner}/{repo}#{pr_number}")
            return False
        
        active_cache = self._get_cache()
        logger.info(f"💾 IntelligentCache.store_general_review: {owner}/{repo}#{pr_number}@{commit_sha[:8]} (backend: {'MongoDB' if self._use_mongo else 'In-memory'})")
        try:
            result = active_cache.store_general_review(owner, repo, pr_number, commit_sha, review_data, metadata)
            if result:
                logger.info(f"✅ IntelligentCache.store_general_review SUCCESS: {owner}/{repo}#{pr_number}@{commit_sha[:8]}")
            else:
                logger.error(f"❌ IntelligentCache.store_general_review FAILED: {owner}/{repo}#{pr_number}@{commit_sha[:8]} - backend returned False")
            return result
        except Exception as e:
            logger.error(f"❌ IntelligentCache.store_general_review EXCEPTION: {owner}/{repo}#{pr_number}@{commit_sha[:8]} - {e}", exc_info=True)
            return False
    
    # Thread Analysis methods
    def get_thread_analysis(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                           thread_id: str, max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached thread analysis"""
        return self._get_cache().get_thread_analysis(owner, repo, pr_number, commit_sha, thread_id, max_age_hours)
    
    def store_thread_analysis(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                              thread_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Store thread analysis in cache"""
        return self._get_cache().store_thread_analysis(owner, repo, pr_number, commit_sha, thread_id, analysis_data)
    
    # Analysis Summary methods
    def get_analysis_summary(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                            max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached analysis summary"""
        return self._get_cache().get_analysis_summary(owner, repo, pr_number, commit_sha, max_age_hours)
    
    def store_analysis_summary(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                              summary_data: Dict[str, Any]) -> bool:
        """Store analysis summary in cache"""
        return self._get_cache().store_analysis_summary(owner, repo, pr_number, commit_sha, summary_data)
    
    # Embedding methods
    def get_embeddings(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                      indexing_paths: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached embeddings.
        
        Returns:
            Dict with 'documents', 'embeddings' (numpy array), 'metadata', or None if not found
        """
        return self._get_cache().get_embeddings(owner, repo, pr_number, commit_sha, indexing_paths)
    
    def store_embeddings(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                        documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]],
                        indexing_paths: Optional[List[str]] = None) -> bool:
        """Store embeddings in cache"""
        return self._get_cache().store_embeddings(owner, repo, pr_number, commit_sha, documents, embeddings, metadata, indexing_paths)
    
    # Embedding Metadata methods
    def get_embedding_metadata(self, owner: str, repo: str, pr_number: int, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get embedding metadata"""
        return self._get_cache().get_embedding_metadata(owner, repo, pr_number, commit_sha)
    
    def store_embedding_metadata(self, owner: str, repo: str, pr_number: int, commit_sha: str,
                                 indexing_paths: Optional[List[str]] = None, total_files: int = 0,
                                 total_chunks: int = 0) -> bool:
        """Store embedding metadata"""
        return self._get_cache().store_embedding_metadata(owner, repo, pr_number, commit_sha, indexing_paths, total_files, total_chunks)
    
    # Utility methods
    def get_commit_freshness(self, owner: str, repo: str, pr_number: int, commit_sha: str) -> Optional[datetime]:
        """Get the freshness timestamp for a commit"""
        cache = self._get_cache()
        if hasattr(cache, 'get_commit_freshness'):
            return cache.get_commit_freshness(owner, repo, pr_number, commit_sha)
        return None
    
    def clear_commit_cache(self, owner: str, repo: str, pr_number: int, commit_sha: str) -> bool:
        """Clear all cached data for a specific commit"""
        cache = self._get_cache()
        if hasattr(cache, 'clear_commit_cache'):
            return cache.clear_commit_cache(owner, repo, pr_number, commit_sha)
        return False
    
    def get_latest_cached_commit(self, owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        """Get the latest cached commit SHA for a PR"""
        cache = self._get_cache()
        if hasattr(cache, 'get_latest_cached_commit'):
            return cache.get_latest_cached_commit(owner, repo, pr_number)
        return None
