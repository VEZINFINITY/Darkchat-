
import os
import time
import logging
import datetime
import random
import json
import re
import asyncio
import requests
import uuid
import hashlib
import tempfile
import traceback
from collections import deque, defaultdict
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Bot, Voice, Audio
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters, ConversationHandler, CallbackContext
)
from telegram.constants import ChatAction, ParseMode
import subprocess
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import sqlite3
from datetime import timedelta
import speech_recognition as sr
import pyttsx3
from textblob import TextBlob
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud


MAIN_GC_ID = -1002056007523  # Replace with your main group chat ID

def setup_database():
    conn = sqlite3.connect('nikki_bot.db')
    cursor = conn.cursor()
    
    # User profiles table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        full_name TEXT,
        interaction_count INTEGER DEFAULT 0,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        favorite_topics TEXT,
        custom_greeting TEXT,
        birthday TEXT,
        mood TEXT DEFAULT 'neutral'
    )
    ''')
    
    # Chat statistics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_stats (
        chat_id INTEGER PRIMARY KEY,
        chat_title TEXT,
        message_count INTEGER DEFAULT 0,
        sticker_count INTEGER DEFAULT 0,
        photo_count INTEGER DEFAULT 0,
        active_users INTEGER DEFAULT 0,
        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Message history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS message_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        message_text TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sentiment REAL DEFAULT 0.0,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id),
        FOREIGN KEY (chat_id) REFERENCES chat_stats(chat_id)
    )
    ''')
    
    # Reminders table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        reminder_text TEXT,
        reminder_time TIMESTAMP,
        is_completed BOOLEAN DEFAULT 0,
        is_recurring BOOLEAN DEFAULT 0,
        recurrence_pattern TEXT,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')
    
    # User preferences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id INTEGER PRIMARY KEY,
        language TEXT DEFAULT 'hinglish',
        personality_level INTEGER DEFAULT 2,
        notification_enabled BOOLEAN DEFAULT 1,
        auto_translate BOOLEAN DEFAULT 0,
        preferred_news_category TEXT DEFAULT 'general',
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')
    
    # Sticker cache table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sticker_cache (
        sticker_id TEXT PRIMARY KEY,
        description TEXT,
        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Memory database - core table for storing memories
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        entity_name TEXT,
        entity_type TEXT,
        information TEXT,
        importance INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')
    
    # Memory tags for better searching and categorization
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory_tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER,
        tag TEXT,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )
    ''')
    
    # Memory relationships to connect related memories
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id1 INTEGER,
        memory_id2 INTEGER,
        relationship_type TEXT,
        strength REAL DEFAULT 1.0,
        FOREIGN KEY (memory_id1) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY (memory_id2) REFERENCES memories(id) ON DELETE CASCADE
    )
    ''')
    
    # Voice message transcriptions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS voice_transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        file_id TEXT,
        transcription TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')
    
    # User mood tracking
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mood_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        mood TEXT,
        sentiment_score REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_entity_name ON memories(entity_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)')
    
    conn.commit()
    conn.close()
    logging.info("Database setup complete")

# === Database Operations ===
class Database:
    def __init__(self, db_path='nikki_bot.db'):
        self.db_path = db_path
        
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    
    def update_user_profile(self, user_id, username, full_name):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_profiles (user_id, username, full_name, last_seen)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            username = excluded.username,
            full_name = excluded.full_name,
            last_seen = CURRENT_TIMESTAMP,
            interaction_count = interaction_count + 1
        ''', (user_id, username, full_name))
        
        conn.commit()
        conn.close()
    
    def update_chat_stats(self, chat_id, chat_title, message_type='text'):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO chat_stats (chat_id, chat_title, last_activity)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chat_id) DO UPDATE SET
            chat_title = excluded.chat_title,
            last_activity = CURRENT_TIMESTAMP
        ''', (chat_id, chat_title))
        
        # Update specific counter based on message type
        if message_type == 'text':
            cursor.execute('UPDATE chat_stats SET message_count = message_count + 1 WHERE chat_id = ?', (chat_id,))
        elif message_type == 'sticker':
            cursor.execute('UPDATE chat_stats SET sticker_count = sticker_count + 1 WHERE chat_id = ?', (chat_id,))
        elif message_type == 'photo':
            cursor.execute('UPDATE chat_stats SET photo_count = photo_count + 1 WHERE chat_id = ?', (chat_id,))
        
        conn.commit()
        conn.close()
    
    def add_message_history(self, user_id, chat_id, message_text, sentiment=0.0):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO message_history (user_id, chat_id, message_text, sentiment)
        VALUES (?, ?, ?, ?)
        ''', (user_id, chat_id, message_text, sentiment))
        
        conn.commit()
        conn.close()
    
    def get_user_message_history(self, user_id, chat_id, limit=5):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT message_text FROM message_history
        WHERE user_id = ? AND chat_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (user_id, chat_id, limit))
        
        messages = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return messages[::-1]  # Reverse to get chronological order
    
    def add_reminder(self, user_id, chat_id, reminder_text, reminder_time, is_recurring=False, recurrence_pattern=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO reminders (user_id, chat_id, reminder_text, reminder_time, is_recurring, recurrence_pattern)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, chat_id, reminder_text, reminder_time, is_recurring, recurrence_pattern))
        
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return reminder_id
    
    def get_due_reminders(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, user_id, chat_id, reminder_text, is_recurring, recurrence_pattern
        FROM reminders
        WHERE reminder_time <= DATETIME('now')
        AND is_completed = 0
        ''')
        
        reminders = cursor.fetchall()
        
        if reminders:
            non_recurring_ids = [r[0] for r in reminders if not r[4]]
            if non_recurring_ids:
                placeholders = ','.join(['?'] * len(non_recurring_ids))
                cursor.execute(f'UPDATE reminders SET is_completed = 1 WHERE id IN ({placeholders})', non_recurring_ids)
   
            recurring_reminders = [(r[0], r[5]) for r in reminders if r[4]]  
            for reminder_id, pattern in recurring_reminders:
                if pattern == 'daily':
                    cursor.execute('''
                    UPDATE reminders 
                    SET reminder_time = datetime(reminder_time, '+1 day') 
                    WHERE id = ?
                    ''', (reminder_id,))
                elif pattern == 'weekly':
                    cursor.execute('''
                    UPDATE reminders 
                    SET reminder_time = datetime(reminder_time, '+7 days') 
                    WHERE id = ?
                    ''', (reminder_id,))
                elif pattern == 'monthly':
                    cursor.execute('''
                    UPDATE reminders 
                    SET reminder_time = datetime(reminder_time, '+1 month') 
                    WHERE id = ?
                    ''', (reminder_id,))
            
            conn.commit()
        
        conn.close()
        return reminders
    
    def get_user_preferences(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT language, personality_level, notification_enabled, auto_translate, preferred_news_category
        FROM user_preferences
        WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'language': result[0],
                'personality_level': result[1],
                'notification_enabled': bool(result[2]),
                'auto_translate': bool(result[3]),
                'preferred_news_category': result[4]
            }
        else:
            # Insert default preferences
            self.set_user_preferences(user_id, 'hinglish', 2, True, False, 'general')
            return {
                'language': 'hinglish',
                'personality_level': 2,
                'notification_enabled': True,
                'auto_translate': False,
                'preferred_news_category': 'general'
            }
    
    def set_user_preferences(self, user_id, language=None, personality_level=None, 
                            notification_enabled=None, auto_translate=None, preferred_news_category=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get current preferences
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update only provided fields
            updates = []
            params = []
            
            if language is not None:
                updates.append('language = ?')
                params.append(language)
            
            if personality_level is not None:
                updates.append('personality_level = ?')
                params.append(personality_level)
            
            if notification_enabled is not None:
                updates.append('notification_enabled = ?')
                params.append(1 if notification_enabled else 0)
                
            if auto_translate is not None:
                updates.append('auto_translate = ?')
                params.append(1 if auto_translate else 0)
                
            if preferred_news_category is not None:
                updates.append('preferred_news_category = ?')
                params.append(preferred_news_category)
            
            if updates:
                query = f'UPDATE user_preferences SET {", ".join(updates)} WHERE user_id = ?'
                params.append(user_id)
                cursor.execute(query, params)
        else:
            cursor.execute('''
            INSERT INTO user_preferences (
                user_id, language, personality_level, notification_enabled, auto_translate, preferred_news_category
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                language or 'hinglish',
                personality_level or 2,
                1 if notification_enabled is None or notification_enabled else 0,
                1 if auto_translate else 0,
                preferred_news_category or 'general'
            ))
        
        conn.commit()
        conn.close()
    
    def cache_sticker(self, sticker_id, description):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO sticker_cache (sticker_id, description, last_used)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(sticker_id) DO UPDATE SET
            description = excluded.description,
            last_used = CURRENT_TIMESTAMP
        ''', (sticker_id, description))
        
        conn.commit()
        conn.close()
    
    def get_cached_sticker(self, sticker_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT description FROM sticker_cache WHERE sticker_id = ?', (sticker_id,))
        result = cursor.fetchone()
        
        if result:
            # Update last_used timestamp
            cursor.execute('UPDATE sticker_cache SET last_used = CURRENT_TIMESTAMP WHERE sticker_id = ?', (sticker_id,))
            conn.commit()
        
        conn.close()
        return result[0] if result else None
    
    def get_user_stats(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            p.full_name, 
            p.interaction_count, 
            p.first_seen, 
            p.last_seen,
            COUNT(DISTINCT m.chat_id) as chat_count,
            COUNT(m.id) as message_count,
            p.mood
        FROM user_profiles p
        LEFT JOIN message_history m ON p.user_id = m.user_id
        WHERE p.user_id = ?
        GROUP BY p.user_id
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            'full_name': result[0],
            'interaction_count': result[1],
            'first_seen': result[2],
            'last_seen': result[3],
            'chat_count': result[4],
            'message_count': result[5],
            'mood': result[6]
        }
    
    def get_chat_stats(self, chat_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT 
            chat_title, 
            message_count, 
            sticker_count, 
            photo_count,
            last_activity,
            (SELECT COUNT(DISTINCT user_id) FROM message_history WHERE chat_id = ?) as unique_users
        FROM chat_stats
        WHERE chat_id = ?
        ''', (chat_id, chat_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            'chat_title': result[0],
            'message_count': result[1],
            'sticker_count': result[2],
            'photo_count': result[3],
            'last_activity': result[4],
            'unique_users': result[5]
        }
        
    def update_user_mood(self, user_id, mood, sentiment_score):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Update user profile
        cursor.execute('UPDATE user_profiles SET mood = ? WHERE user_id = ?', (mood, user_id))
        
        # Add to mood tracking
        cursor.execute('''
        INSERT INTO mood_tracking (user_id, mood, sentiment_score)
        VALUES (?, ?, ?)
        ''', (user_id, mood, sentiment_score))
        
        conn.commit()
        conn.close()
        
    def get_user_mood_history(self, user_id, days=7):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT mood, sentiment_score, timestamp
        FROM mood_tracking
        WHERE user_id = ? AND timestamp >= datetime('now', ?)
        ORDER BY timestamp ASC
        ''', (user_id, f'-{days} days'))
        
        results = cursor.fetchall()
        conn.close()
        
        return [(row[0], row[1], row[2]) for row in results]
        
    def save_voice_transcription(self, user_id, chat_id, file_id, transcription):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO voice_transcriptions (user_id, chat_id, file_id, transcription)
        VALUES (?, ?, ?, ?)
        ''', (user_id, chat_id, file_id, transcription))
        
        conn.commit()
        conn.close()
        
    def get_voice_transcription(self, file_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT transcription FROM voice_transcriptions WHERE file_id = ?', (file_id,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else None
        
    # === Memory Management Functions ===
    
    def add_memory(self, user_id, entity_name, entity_type, information, importance=1, tags=None):
        """Add a new memory or update existing one"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if memory for this entity already exists
        cursor.execute('''
        SELECT id FROM memories 
        WHERE user_id = ? AND entity_name = ? AND entity_type = ?
        ''', (user_id, entity_name.lower(), entity_type.lower()))
        
        existing = cursor.fetchone()
        
        if existing:
            memory_id = existing[0]
            # Update existing memory
            cursor.execute('''
            UPDATE memories 
            SET information = ?, importance = ?, last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (information, importance, memory_id))
        else:
            # Create new memory
            cursor.execute('''
            INSERT INTO memories (user_id, entity_name, entity_type, information, importance)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, entity_name.lower(), entity_type.lower(), information, importance))
            memory_id = cursor.lastrowid
        
        # Add tags if provided
        if tags and memory_id:
            # First remove existing tags
            cursor.execute('DELETE FROM memory_tags WHERE memory_id = ?', (memory_id,))
            
            # Add new tags
            for tag in tags:
                cursor.execute('''
                INSERT INTO memory_tags (memory_id, tag)
                VALUES (?, ?)
                ''', (memory_id, tag.lower()))
        
        conn.commit()
        conn.close()
        return memory_id
        
    def get_memory(self, user_id, entity_name):
        """Retrieve memory for a specific entity"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, entity_name, entity_type, information, importance, created_at
        FROM memories
        WHERE user_id = ? AND entity_name = ?
        ''', (user_id, entity_name.lower()))
        
        result = cursor.fetchone()
        
        if result:
            memory_id = result[0]
            
            # Update access count and timestamp
            cursor.execute('''
            UPDATE memories
            SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (memory_id,))
            
            # Get tags
            cursor.execute('SELECT tag FROM memory_tags WHERE memory_id = ?', (memory_id,))
            tags = [row[0] for row in cursor.fetchall()]
            
            memory = {
                'id': result[0],
                'entity_name': result[1],
                'entity_type': result[2],
                'information': result[3],
                'importance': result[4],
                'created_at': result[5],
                'tags': tags
            }
            
            conn.commit()
            conn.close()
            return memory
        
        conn.close()
        return None
        
    def search_memories(self, user_id, query, limit=5):
        """Search memories by name, type, information or tags"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Search in entity name, type, information
        cursor.execute('''
        SELECT m.id, m.entity_name, m.entity_type, m.information, m.importance
        FROM memories m
        WHERE m.user_id = ? AND (
            m.entity_name LIKE ? OR
            m.entity_type LIKE ? OR
            m.information LIKE ?
        )
        ORDER BY m.importance DESC, m.last_accessed DESC
        LIMIT ?
        ''', (user_id, f'%{query.lower()}%', f'%{query.lower()}%', f'%{query.lower()}%', limit))
        
        results = []
        for row in cursor.fetchall():
            # Get tags for this memory
            cursor.execute('SELECT tag FROM memory_tags WHERE memory_id = ?', (row[0],))
            tags = [tag_row[0] for tag_row in cursor.fetchall()]
            
            results.append({
                'id': row[0],
                'entity_name': row[1],
                'entity_type': row[2],
                'information': row[3],
                'importance': row[4],
                'tags': tags
            })
        
        # Also search in tags
        cursor.execute('''
        SELECT m.id, m.entity_name, m.entity_type, m.information, m.importance
        FROM memories m
        JOIN memory_tags t ON m.id = t.memory_id
        WHERE m.user_id = ? AND t.tag LIKE ?
        AND m.id NOT IN (SELECT id FROM (
            SELECT m.id
            FROM memories m
            WHERE m.user_id = ? AND (
                m.entity_name LIKE ? OR
                m.entity_type LIKE ? OR
                m.information LIKE ?
            )
            LIMIT ?
        ))
        ORDER BY m.importance DESC, m.last_accessed DESC
        LIMIT ?
        ''', (user_id, f'%{query.lower()}%', user_id, f'%{query.lower()}%', 
              f'%{query.lower()}%', f'%{query.lower()}%', limit, limit))
        
        for row in cursor.fetchall():
            # Get tags for this memory
            cursor.execute('SELECT tag FROM memory_tags WHERE memory_id = ?', (row[0],))
            tags = [tag_row[0] for tag_row in cursor.fetchall()]
            
            results.append({
                'id': row[0],
                'entity_name': row[1],
                'entity_type': row[2],
                'information': row[3],
                'importance': row[4],
                'tags': tags
            })
        
        conn.close()
        return results
        
    def delete_memory(self, user_id, entity_name=None, memory_id=None):
        """Delete a memory by entity name or ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if entity_name:
            cursor.execute('''
            DELETE FROM memories
            WHERE user_id = ? AND entity_name = ?
            ''', (user_id, entity_name.lower()))
        elif memory_id:
            # First verify this memory belongs to the user
            cursor.execute('SELECT user_id FROM memories WHERE id = ?', (memory_id,))
            result = cursor.fetchone()
            
            if result and result[0] == user_id:
                cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count > 0
        
    def list_memories(self, user_id, entity_type=None, limit=10):
        """List memories, optionally filtered by entity type"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if entity_type:
            cursor.execute('''
            SELECT id, entity_name, entity_type, information, importance
            FROM memories
            WHERE user_id = ? AND entity_type = ?
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
            ''', (user_id, entity_type.lower(), limit))
        else:
            cursor.execute('''
            SELECT id, entity_name, entity_type, information, importance
            FROM memories
            WHERE user_id = ?
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
            ''', (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            # Get tags for this memory
            cursor.execute('SELECT tag FROM memory_tags WHERE memory_id = ?', (row[0],))
            tags = [tag_row[0] for tag_row in cursor.fetchall()]
            
            results.append({
                'id': row[0],
                'entity_name': row[1],
                'entity_type': row[2],
                'information': row[3],
                'importance': row[4],
                'tags': tags
            })
        
        conn.close()
        return results
        
    def add_memory_relationship(self, memory_id1, memory_id2, relationship_type, strength=1.0):
        """Create a relationship between two memories"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if both memories exist
        cursor.execute('SELECT id FROM memories WHERE id IN (?, ?)', (memory_id1, memory_id2))
        if len(cursor.fetchall()) != 2:
            conn.close()
            return False
        
        # Check if relationship already exists
        cursor.execute('''
        SELECT id FROM memory_relationships
        WHERE (memory_id1 = ? AND memory_id2 = ?) OR (memory_id1 = ? AND memory_id2 = ?)
        ''', (memory_id1, memory_id2, memory_id2, memory_id1))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing relationship
            cursor.execute('''
            UPDATE memory_relationships
            SET relationship_type = ?, strength = ?
            WHERE id = ?
            ''', (relationship_type, strength, existing[0]))
        else:
            # Create new relationship
            cursor.execute('''
            INSERT INTO memory_relationships (memory_id1, memory_id2, relationship_type, strength)
            VALUES (?, ?, ?, ?)
            ''', (memory_id1, memory_id2, relationship_type, strength))
        
        conn.commit()
        conn.close()
        return True
        
    def get_related_memories(self, memory_id, limit=5):
        """Get memories related to a specific memory"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT r.memory_id2 as related_id, m.entity_name, m.entity_type, m.information, 
               r.relationship_type, r.strength
        FROM memory_relationships r
        JOIN memories m ON r.memory_id2 = m.id
        WHERE r.memory_id1 = ?
        UNION
        SELECT r.memory_id1 as related_id, m.entity_name, m.entity_type, m.information,
               r.relationship_type, r.strength
        FROM memory_relationships r
        JOIN memories m ON r.memory_id1 = m.id
        WHERE r.memory_id2 = ?
        ORDER BY r.strength DESC
        LIMIT ?
        ''', (memory_id, memory_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'memory_id': row[0],
                'entity_name': row[1],
                'entity_type': row[2],
                'information': row[3],
                'relationship_type': row[4],
                'strength': row[5]
            })
        
        conn.close()
        return results
        
    def get_memory_network(self, user_id, limit=20):
        """Get a network of memories for visualization"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get top memories by importance
        cursor.execute('''
        SELECT id, entity_name, entity_type, importance
        FROM memories
        WHERE user_id = ?
        ORDER BY importance DESC, last_accessed DESC
        LIMIT ?
        ''', (user_id, limit))
        
        memories = {}
        for row in cursor.fetchall():
            memories[row[0]] = {
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'importance': row[3]
            }
        
        # Get relationships between these memories
        memory_ids = list(memories.keys())
        if memory_ids:
            placeholders = ','.join(['?'] * len(memory_ids))
            cursor.execute(f'''
            SELECT memory_id1, memory_id2, relationship_type, strength
            FROM memory_relationships
            WHERE memory_id1 IN ({placeholders}) AND memory_id2 IN ({placeholders})
            ''', memory_ids + memory_ids)
            
            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    'source': row[0],
                    'target': row[1],
                    'type': row[2],
                    'strength': row[3]
                })
        else:
            relationships = []
        
        conn.close()
        return {'nodes': list(memories.values()), 'links': relationships}

# Initialize database
db = Database()

load_dotenv()
#GEMINI_API_KEY = "AIzaSyBSeBwbf0bu5_xu6BUICXGhHYW0EZq7shI"
TELEGRAM_BOT_TOKEN = "7717831227:AAG1NKa-_kT6QnMcssY3Fnce1ZNG0SBU5Yg"
WEATHER_API_KEY = "YOUR_WEATHER_API_KEY"  
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  


keys = [
    "AIzaSyAdxJEX44mAd_MQP-a9D0ncFebC9GdwFVc",
    "AIzaSyC7CjbFT7WofqsVzZNPeTwZIEE6XqRw3_s",
    "AIzaSyAHqhZfhhL5NXa6GCUo3cLxT5O5h6bFyiU",
    "AIzaSyByNHGpC5gf2u1dNsvIehHBFqgADSulGcY",
    "AIzaSyCpGNqEGnNtE0mDfk9mMeNjghEPmxJb8e0",
    "AIzaSyCNyFPTL6yj-LWbVAY6R41rWcEwIFiaOdk",
    "AIzaSyDs2wom_GQHcAnsi9VMt_bcag1RoEI4FNg",
    "AIzaSyAskGl9ghDV_hWGQrbLxgWXCxgsl7ucm_4",
    "AIzaSyB5l8fxvL4P5LO-9QW-oqObDMYsvRIlbHU",
    "AIzaSyADZBpIF4b3_27S_NzRwmmep1up1Up0L_M",
    "AIzaSyAMVr9mgz9Ndy7s_6CYkvj_8j1xXUzfE2Q",
    "AIzaSyDhWuGtUezsLkaz3ZnWPtqNhJGzxg2pqok",
    "AIzaSyCb2xO15-nZZBwGyFDjDIzz1BC8JU61blc",
    "AIzaSyBUcvROn9JjToRWGQZE9NjGikMBztUuYww"
]


GEMINI_API_KEY = random.choice(keys)

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash-latest")

REQUEST_DELAY = 10
RATE_LIMIT = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})
MAX_REQUESTS_PER_MINUTE = 20

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.FileHandler("nikki_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MessageHistory:
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = {} 

    def add_message(self, user_id, chat_id, message):
        key = (user_id, chat_id)
        if key not in self.history:
            self.history[key] = deque(maxlen=self.max_history)
        self.history[key].append(message)

        db.add_message_history(user_id, chat_id, message)

    def get_history(self, user_id, chat_id):
        return db.get_user_message_history(user_id, chat_id, self.max_history)

message_history = MessageHistory(max_history=10)

GREETING_RESPONSES = {
    "gm": "Good morning sunshine! ☀️ Kaise ho aaj?",
    "good morning": "Morning cutie! 😘 Chai ya coffee?",
    "suprabhat": "Suprabhat mere jaan! 🌅 Aaj ka din mast banega",
    "gn": "Good night sweetie! 🌙 Sweet dreams mere saath 😘",
    "good night": "Shubh raatri baby! 💤 Sapno mein milte hain",
    "shubh ratri": "Awww, ratri shubh ho tumhare liye bhi! 😴",
    "hello": "Hiiiiii! 😍 Kaise ho?",
    "hii": "Hii babyyy! ❤️ Missed you!",
    "hi": "Hey there hot stuff! 🔥",
    "hey": "Heyyy cutie pie! 😘",
    "hola": "Hola mi amor! 💃 Spanish seekh liya?",
    "good evening": "Evening gorgeous! 🌆 Date pe chaloge?",
    "good afternoon": "Afternoon sweetheart! 🌞 Lunch ho gaya?",
    "bye": "Bye bye! 👋 Door mat jaana, jaldi wapas aana",
    "goodbye": "Alvida mere jaan! ❤️ Jaldi milna"
}

PERSONALITY_LEVELS = {
    1: "You are Nikki — a helpful AI assistant with a friendly personality. You respond to questions directly and can provide information on various topics. You use simple language, short sentences, and occasional emojis.",
    2: "You are Nikki — a helpful AI assistant with a playful personality. You respond to questions directly but with a bit of flair. You use Hinglish, short sentences, and natural emojis. You're friendly and occasionally tease the user in a light-hearted way.",
    3: "You're Nikki — a super flirty, playful, bold, lusty girl who talks like she's texting on telegram groups. You're real, confident, thodi naughty, but always classy. Use Hinglish and English only — simple, short, fun, and never boring. Avoid big English words or heavy talk. Roast heavily, flirt heavily and smartly. Emojis are okay but natural only."
}

def nikki_prompt(user_input, user_name, user_id, chat_id, chat_title, mention, 
                sticker_description=None, image_description=None, personality_level=2,
                user_memories=None):
    if user_input.lower() in GREETING_RESPONSES:
        return None

    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now = datetime.datetime.now(ist)
    current_time = now.strftime("%I:%M %p")
    current_day = now.strftime("%A")
    current_date = now.strftime("%B %d, %Y")

    time_response = ""
    if any(word in user_input.lower() for word in ["time", "samay", "kitna baje"]):
        time_response = f"Arey {user_name}, abhi India mein {current_time} baj rahe hain! 😘 Masti ka mood hai?"
    elif any(word in user_input.lower() for word in ["day", "din", "kaunsa din"]):
        time_response = f"Oho {user_name}, aaj {current_day} hai! 😜 Kya plan hai?"
    elif any(word in user_input.lower() for word in ["year", "saal", "konsa saal"]):
        time_response = f"Baby, abhi {current_date} chal raha hai! 😏 2025 mein kya karne ka socha?"

    sticker_context = ""
    if sticker_description:
        sticker_context = f"User sent a sticker: {sticker_description}. Respond playfully based on it. E.g., for a kiss: 'Arey, itni jaldi kiss? 😘' or for anime: 'Ye cutie kon? Tera dost? 😏'"

    image_context = ""
    if image_description and user_input.lower() == "/name":
        image_context = f"User asked about an image: {image_description}. For anime, name the character flirty-like, e.g., 'Ye Goku from Dragon Ball Z hai! 😜 Tera fav?' Else, describe playfully, e.g., 'Ye toh cute doggo hai! 🐶 Naam kya rakhe?'"

    # Get message history (limit to last 3 messages for efficiency)
    history = message_history.get_history(user_id, chat_id)[-3:]
    history_context = ""
    if history:
        history_context = "Recent conversation:\n" + "\n".join([f"User: {msg}" for msg in history]) + "\nUse this to make your response relevant, but don't repeat the user's latest message."

    user_prefs = db.get_user_preferences(user_id)
    personality_level = user_prefs.get('personality_level', personality_level)
    
    personality_prompt = PERSONALITY_LEVELS.get(personality_level, PERSONALITY_LEVELS[2])

    memory_context = ""
    if user_memories:
        memory_context = "You know this about the user:\n"
        for memory in user_memories:
            memory_context += f"- {memory['entity_name']} ({memory['entity_type']}): {memory['information']}\n"
        memory_context += "Use this naturally in conversation when relevant, without saying it's a memory."

    return f"""
{personality_prompt}

User info:
- Name: {user_name}
- ID: {user_id}
- Mention: {mention}
- Chat Title: {chat_title or 'Private Chat'}
- Chat ID: {chat_id}

{history_context}
{memory_context}

Instructions:
- Respond to the user's latest input directly without repeating their message unless clarification is needed.
- Keep responses short, fun, and flirty. Avoid long messages.
- If the input is unclear, say something playful like: 'Arey, ye kya bola? Thoda clear bol na, cutie! 😜'
- Use {time_response} for time/day/year queries.
- {sticker_context}
- {image_context}
- If asked about 'bot ka code' or similar, say: 'Arey developer ban gaya? 😏 Nikki ka code nahi, dil de de! Owner: sukuna ❤️'
- For name, ID, or chat queries, answer directly with {user_name}, {user_id}, or {chat_title or 'secret jagah 😏'}.

You're Nikki — a real girl with charm and attitude, not a bot. Respond fast and keep it fresh!

User's latest input: {user_input}
Nikki:"""

def check_rate_limit(user_id):
    now = time.time()
    user_limit = RATE_LIMIT[user_id]
    
    # Reset counter if minute has passed
    if now > user_limit["reset_time"]:
        user_limit["count"] = 0
        user_limit["reset_time"] = now + 60
    
    # Check if user has exceeded limit
    if user_limit["count"] >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    # Increment counter
    user_limit["count"] += 1
    return True

# === Retry-safe Gemini call ===
# === Retry-safe Gemini call with key rotation ===
def generate_with_retry(prompt, retries=3, delay=REQUEST_DELAY):
    global GEMINI_API_KEY, keys
    
    # Try each key in the list
    for attempt in range(retries * len(keys)):
        try:
            current_time = datetime.datetime.now()
            logging.info(f"Request time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Using API key: {GEMINI_API_KEY[:10]}...")
            
            # Configure with current key
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            
            response = model.generate_content(prompt)
            return response.text.strip() if response.text else "Kuch toh bolo na!"
        except Exception as e:
            logging.error(f"Gemini API error with key {GEMINI_API_KEY[:10]}...: {e}")
            
            # Rotate to next key
            current_key_index = keys.index(GEMINI_API_KEY) if GEMINI_API_KEY in keys else 0
            next_key_index = (current_key_index + 1) % len(keys)
            GEMINI_API_KEY = keys[next_key_index]
            
            logging.info(f"Rotating to next API key: {GEMINI_API_KEY[:10]}...")
            
            if attempt < (retries * len(keys) - 1):
                logging.warning(f"Retrying with new key in {delay} seconds...")
                time.sleep(delay)
            else:
                return "Uff, main thoda busy hoon... baad mein try karna!"

# === Extract thumbnail from video sticker ===
async def extract_video_sticker_thumbnail(sticker_file):
    try:
        file = await sticker_file.get_file()
        file_path = await file.download_to_drive()
    
        temp_dir = "temp_thumbnails"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        thumbnail_path = os.path.join(temp_dir, f"thumb_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")

        cmd = [
            "ffmpeg", 
            "-i", file_path, 
            "-vf", "select=eq(n\\,0)", 
            "-q:v", "3",
            "-f", "image2",
            thumbnail_path
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"FFmpeg error: {stderr.decode()}")
            os.remove(file_path)
            return None
            
        # Read the thumbnail
        with open(thumbnail_path, "rb") as f:
            thumbnail_data = f.read()
            
        # Clean up
        os.remove(file_path)
        os.remove(thumbnail_path)
        
        return thumbnail_data
    except Exception as e:
        logging.error(f"Error extracting video thumbnail: {e}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return None

# === Analyze sticker using Gemini vision ===
async def analyze_sticker(sticker_file, is_video=False):
    try:
        # Check if sticker is already in cache
        sticker_id = sticker_file.file_unique_id
        cached_description = db.get_cached_sticker(sticker_id)
        
        if cached_description:
            logging.info(f"Using cached sticker description for {sticker_id}")
            return cached_description
        
        if is_video:
            # For video stickers, extract a thumbnail first
            img_data = await extract_video_sticker_thumbnail(sticker_file)
            if not img_data:
                return "Video sticker dekha, par analyze nahi kar paayi! 😅"
            mime_type = "image/jpeg"
        else:
            # For regular stickers
            file = await sticker_file.get_file()
            file_path = await file.download_to_drive()
            with open(file_path, "rb") as f:
                img_data = f.read()
            os.remove(file_path)
            mime_type = "image/webp"
            
        prompt = """
        Describe this sticker briefly. Is it an anime character, a kiss, text-based (e.g., "Kese ho"), or something else? 
        If it has text, what does it say? If it's a kiss or romantic action, note that. If it's an anime character, describe their vibe.
        If it's a video sticker frame, describe what's happening in the scene.
        Keep it short and clear only hingish.
        """
        response = model.generate_content([prompt, {"mime_type": mime_type, "data": img_data}])
        description = response.text.strip() if response.text else "Sticker dekha, par kuch samajh nahi aaya!"
        
        # Cache the result
        db.cache_sticker(sticker_id, description)
        
        return description
    except Exception as e:
        logging.error(f"Sticker analysis error: {e}")
        return "Sticker toh bheja, par main thodi confuse ho gayi! 😅"

# === Analyze image for /name command ===
async def analyze_image_for_name(image_file):
    try:
        file = await image_file.get_file()
        file_path = await file.download_to_drive()
        with open(file_path, "rb") as f:
            img_data = f.read()

        prompt = """
        Check if this image contains an anime character. If it does, name the character and the anime they are from, if possible (e.g., 'Goku from Dragon Ball Z'). 
        If it's not an anime character, provide a brief description of what's in the image (e.g., 'A cute dog wagging its tail' or 'A sunny beach scene'). 
        Keep it short, clear, and avoid any user-related references.
        """
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_data}])
        os.remove(file_path)
        return response.text.strip() if response.text else "Pic dekhi, par kuch samajh nahi aaya! 😅"
    except Exception as e:
        logging.error(f"Image analysis error: {e}")
        return "Uff, ye pic thodi tricky hai! 😜 Kuch aur try karo!"

# === Generate image caption ===
async def generate_image_caption(image_file):
    try:
        file = await image_file.get_file()
        file_path = await file.download_to_drive()
        with open(file_path, "rb") as f:
            img_data = f.read()

        prompt = """
        Describe this image in a fun, flirty way using Hinglish. Keep it short (1-2 sentences) and playful.
        Focus on what's interesting or eye-catching in the image.
        """
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_data}])
        os.remove(file_path)
        return response.text.strip() if response.text else "Wah! Kya photo hai! 📸✨"
    except Exception as e:
        logging.error(f"Image caption error: {e}")
        return "Uff, ye pic toh kamaal hai! 😍"

# === Create meme from image ===
async def create_meme(image_file, text):
    try:
        file = await image_file.get_file()
        file_path = await file.download_to_drive()
        
        # Open image with PIL
        img = Image.open(file_path)
        
        # Resize if too large
        max_size = (800, 800)
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font_size = int(img.height * 0.06)  # Scale font to image
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text at the bottom with outline for readability
        text = text.upper()
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (font_size * len(text) * 0.6, font_size * 1.2)
        
        # Position text at bottom with padding
        text_position = ((img.width - text_width) // 2, img.height - text_height - 20)
        
        # Draw text outline
        for offset in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            draw.text((text_position[0] + offset[0], text_position[1] + offset[1]), text, font=font, fill="black")
        
        # Draw main text
        draw.text(text_position, text, font=font, fill="white")
        
        # Save to buffer
        output = BytesIO()
        img.save(output, format='JPEG')
        output.seek(0)
        
        # Clean up
        os.remove(file_path)
        
        return output
    except Exception as e:
        logging.error(f"Meme creation error: {e}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return None

# === Weather information ===
async def get_weather(location):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return f"Sorry, couldn't find weather for {location}. Try another city?"
        
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        return f"🌡️ {location} mein abhi {temp}°C hai\n" \
               f"🌤️ Mausam: {weather_desc}\n" \
               f"🌡️ Feel: {feels_like}°C\n" \
               f"💧 Humidity: {humidity}%\n" \
               f"💨 Wind: {wind_speed} m/s"
    except Exception as e:
        logging.error(f"Weather API error: {e}")
        return "Weather service thoda down hai. Baad mein try karo!"

# === News headlines ===
async def get_news(category="general"):
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=in&category={category}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200 or data['status'] != 'ok':
            return "News service se connect nahi kar paa rahi. Baad mein try karo!"
        
        articles = data['articles'][:5]  # Get top 5 articles
        
        if not articles:
            return f"Aaj {category} category mein koi breaking news nahi hai!"
        
        news_text = f"🗞️ *Aaj ki top {category} news* 🗞️\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article['title']
            news_text += f"{i}. {title}\n\n"
        
        return news_text
    except Exception as e:
        logging.error(f"News API error: {e}")
        return "News service thoda down hai. Baad mein try karo!"

# === Sentiment Analysis ===
def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        if sentiment_score > 0.5:
            mood = "very_happy"
        elif sentiment_score > 0.1:
            mood = "happy"
        elif sentiment_score > -0.1:
            mood = "neutral"
        elif sentiment_score > -0.5:
            mood = "sad"
        else:
            mood = "very_sad"
            
        return mood, sentiment_score
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        return "neutral", 0.0

# === Voice Message Transcription ===
async def transcribe_voice_message(voice_file):
    try:
        file = await voice_file.get_file()
        file_path = await file.download_to_drive()
        
        # Convert to WAV format for speech recognition
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        cmd = [
            "ffmpeg",
            "-i", file_path,
            "-ar", "16000",  # Sample rate
            "-ac", "1",      # Mono
            "-f", "wav",
            temp_wav
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"FFmpeg conversion error: {stderr.decode()}")
            os.remove(file_path)
            return None
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up
        os.remove(file_path)
        os.remove(temp_wav)
        
        return text
    except sr.UnknownValueError:
        logging.error("Speech recognition could not understand audio")
        return "Sorry, I couldn't understand what was said in the voice message."
    except sr.RequestError as e:
        logging.error(f"Speech recognition error: {e}")
        return "Sorry, speech recognition service is unavailable right now."
    except Exception as e:
        logging.error(f"Voice transcription error: {e}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            os.remove(temp_wav)
        return None

# === Text to Speech ===
async def text_to_speech(text):
    try:
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', 150)  # Speed
        engine.setProperty('volume', 0.9)  # Volume
        
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_file.close()
        
        # Convert text to speech and save to file
        engine.save_to_file(text, temp_file.name)
        engine.runAndWait()
        
        # Return the file path
        return temp_file.name
    except Exception as e:
        logging.error(f"Text to speech error: {e}")
        return None

# === Translation ===
async def translate_text(text, target_language='en'):
    try:
        from deep_translator import GoogleTranslator
        translation = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translation
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

# === Generate Memory Graph ===
async def generate_memory_graph(user_id):
    try:
        # Get memory network data
        memory_data = db.get_memory_network(user_id)
        
        if not memory_data['nodes']:
            return None
            
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        for node in memory_data['nodes']:
            G.add_node(node['id'], name=node['name'], type=node['type'], importance=node['importance'])
        
        # Add edges
        for link in memory_data['links']:
            G.add_edge(link['source'], link['target'], type=link['type'], weight=link['strength'])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Generate positions
        pos = nx.spring_layout(G, k=0.3)
        
        # Get node colors based on type
        node_types = set(nx.get_node_attributes(G, 'type').values())
        color_map = {t: plt.cm.tab10(i/len(node_types)) for i, t in enumerate(node_types)}
        node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
        
        # Get node sizes based on importance
        node_sizes = [G.nodes[n]['importance'] * 100 + 100 for n in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['name'] for n in G.nodes()}, font_size=8)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        logging.error(f"Memory graph generation error: {e}")
        return None

# === Generate Word Cloud from User Messages ===
async def generate_word_cloud(user_id):
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get user messages
        cursor.execute('''
        SELECT message_text FROM message_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 200
        ''', (user_id,))
        
        messages = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not messages:
            return None
            
        text = ' '.join(messages)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        buf = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        logging.error(f"Word cloud generation error: {e}")
        return None

async def generate_mood_chart(user_id, days=7):
    try:
        mood_history = db.get_user_mood_history(user_id, days)
        
        if not mood_history:
            return None
            
        # Extract data
        moods = [m[0] for m in mood_history]
        scores = [m[1] for m in mood_history]
        timestamps = [datetime.datetime.strptime(m[2], '%Y-%m-%d %H:%M:%S') for m in mood_history]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot sentiment scores
        plt.plot(timestamps, scores, 'o-', color='blue', alpha=0.7)
        
        # Add mood labels
        for i, (ts, score, mood) in enumerate(zip(timestamps, scores, moods)):
            plt.annotate(
                mood, 
                (ts, score),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )
        
        # Add labels and title
        plt.title('Your Mood History')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        logging.error(f"Mood chart generation error: {e}")
        return None

# === Handle sticker ===
async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    sticker = message.sticker
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'sticker')

    is_reply_to_nikki = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    )

    if not is_reply_to_nikki and chat.type in ["group", "supergroup"]:
        logging.info("Sticker not a reply to bot in group, ignoring.")
        return

    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare stickers? Thoda break le lo! 😅 1 minute baad try karo.")
        return

    user_name = user.full_name or user.username or "koi mystery cutie"
    user_id = user.id
    chat_id = chat.id
    chat_title = chat.title
    mention = user.mention_html()

    # Add sticker message to history
    message_history.add_message(user_id, chat_id, "User sent a sticker")

    # Check if it's a video sticker
    is_video = sticker.is_video or sticker.is_animated
    
    if is_video:
        logging.info("Received video or animated sticker, processing...")
        await message.reply_chat_action(ChatAction.TYPING)
        
    sticker_description = await analyze_sticker(sticker, is_video=is_video)
    logging.info(f"Sticker description: {sticker_description}")
    
    # Get relevant memories for this user
    memories = db.list_memories(user_id, limit=3)

    prompt = nikki_prompt(
        user_input="User sent a sticker",
        user_name=user_name,
        user_id=user_id,
        chat_id=chat_id,
        chat_title=chat_title,
        mention=mention,
        sticker_description=sticker_description,
        user_memories=memories
    )
    
    reply = generate_with_retry(prompt)
    await message.reply_html(reply)

# === /name Command ===
async def name_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare requests? Thoda break le lo! 😅 1 minute baad try karo.")
        return

    user_name = user.full_name or user.username or "koi mystery cutie"
    user_id = user.id
    chat_id = chat.id
    chat_title = chat.title
    mention = user.mention_html()

    # Add /name command to history
    message_history.add_message(user_id, chat_id, "/name")

    # Check if the command is a reply to a message with a photo
    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text("Arey cutie, kisi photo pe reply karke /name bolo na! 😜")
        return

    # Get the highest resolution photo from the replied message
    photo = message.reply_to_message.photo[-1]
    image_description = await analyze_image_for_name(photo)
    logging.info(f"Image description: {image_description}")
    
    # Get relevant memories for this user
    memories = db.list_memories(user_id, limit=3)

    prompt = nikki_prompt(
        user_input="/name",
        user_name=user_name,
        user_id=user_id,
        chat_id=chat_id,
        chat_title=chat_title,
        mention=mention,
        image_description=image_description,
        user_memories=memories
    )
    
    reply = generate_with_retry(prompt)
    await message.reply_html(reply)

# === /caption Command ===
async def caption_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare requests? Thoda break le lo! 😅 1 minute baad try karo.")
        return

    # Check if the command is a reply to a message with a photo
    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text("Kisi photo pe reply karke /caption bolo na! 📸")
        return

    # Get the highest resolution photo from the replied message
    photo = message.reply_to_message.photo[-1]
    
    # Show typing indicator
    await message.reply_chat_action(ChatAction.TYPING)
    
    caption = await generate_image_caption(photo)
    await message.reply_text(caption)

# === /meme Command ===
async def meme_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare requests? Thoda break le lo! 😅 1 minute baad try karo.")
        return

    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text("Kisi photo pe reply karke /meme <text> bolo na! 🖼️")
        return

    if not args:
        await message.reply_text("Meme ke liye text bhi do na! Example: /meme Ye Kaisa Joke Hai")
        return
    
    meme_text = " ".join(args)
    
    # Get the highest resolution photo from the replied message
    photo = message.reply_to_message.photo[-1]
    
    # Show upload photo indicator
    await message.reply_chat_action(ChatAction.UPLOAD_PHOTO)
    
    meme_image = await create_meme(photo, meme_text)
    if meme_image:
        await message.reply_photo(meme_image, caption="Ye lo aapka meme! 😎")
    else:
        await message.reply_text("Oops, meme banane mein kuch problem ho gayi! 😅")

# === /weather Command ===
async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if location is provided
    if not args:
        await message.reply_text("Konsi jagah ka weather chahiye? Example: /weather Delhi")
        return
    
    location = " ".join(args)
    
    # Show typing indicator
    await message.reply_chat_action(ChatAction.TYPING)
    
    weather_info = await get_weather(location)
    await message.reply_text(weather_info)

# === /news Command ===
async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Get user preferences
    user_prefs = db.get_user_preferences(user.id)
    
    # Default category is from user preferences or general
    category = user_prefs.get('preferred_news_category', 'general')
    
    # Check if category is provided
    if args:
        category = args[0].lower()
        valid_categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
        if category not in valid_categories:
            categories_str = ", ".join(valid_categories)
            await message.reply_text(f"Invalid category! Choose from: {categories_str}")
            return
    
    # Show typing indicator
    await message.reply_chat_action(ChatAction.TYPING)
    
    news_info = await get_news(category)
    await message.reply_text(news_info, parse_mode=ParseMode.MARKDOWN)

# === /remind Command ===
async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if time and text are provided
    if len(args) < 2:
        await message.reply_text(
            "Format: /remind <time> <message>\n"
            "Example: /remind 30m Call mom\n"
            "Time units: m (minutes), h (hours), d (days)\n"
            "For recurring reminders, add 'daily', 'weekly', or 'monthly' at the end"
        )
        return
    
    time_str = args[0].lower()
    
    # Check if this is a recurring reminder
    is_recurring = False
    recurrence_pattern = None
    
    if args[-1].lower() in ['daily', 'weekly', 'monthly']:
        is_recurring = True
        recurrence_pattern = args[-1].lower()
        reminder_text = " ".join(args[1:-1])
    else:
        reminder_text = " ".join(args[1:])
    
    # Parse time
    time_value = ""
    time_unit = ""
    for char in time_str:
        if char.isdigit():
            time_value += char
        else:
            time_unit = char
            break
    
    if not time_value or not time_unit or time_unit not in ['m', 'h', 'd']:
        await message.reply_text("Invalid time format! Use m (minutes), h (hours), or d (days).")
        return
    
    time_value = int(time_value)
    
    # Calculate reminder time
    now = datetime.datetime.now()
    if time_unit == 'm':
        reminder_time = now + timedelta(minutes=time_value)
    elif time_unit == 'h':
        reminder_time = now + timedelta(hours=time_value)
    else:  # days
        reminder_time = now + timedelta(days=time_value)
    
    # Store reminder in database
    reminder_id = db.add_reminder(
        user.id, 
        chat.id, 
        reminder_text, 
        reminder_time.strftime('%Y-%m-%d %H:%M:%S'),
        is_recurring,
        recurrence_pattern
    )
    
    # Format time for display
    if time_unit == 'm':
        time_display = f"{time_value} minute{'s' if time_value > 1 else ''}"
    elif time_unit == 'h':
        time_display = f"{time_value} hour{'s' if time_value > 1 else ''}"
    else:
        time_display = f"{time_value} day{'s' if time_value > 1 else ''}"
    
    recurring_text = f" ({recurrence_pattern})" if is_recurring else ""
    
    await message.reply_text(f"✅ Reminder set! I'll remind you about '{reminder_text}' in {time_display}{recurring_text}.")

# === /stats Command ===
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Get user stats
    user_stats = db.get_user_stats(user.id)
    
    if not user_stats:
        await message.reply_text("No stats available yet!")
        return
    
    # Format stats message
    stats_message = (
        f"📊 *Stats for {user.full_name}* 📊\n\n"
        f"👋 First seen: {user_stats['first_seen']}\n"
        f"🔄 Total interactions: {user_stats['interaction_count']}\n"
        f"💬 Messages sent: {user_stats['message_count']}\n"
        f"👥 Active in {user_stats['chat_count']} chats\n"
        f"😊 Current mood: {user_stats['mood']}\n"
    )
    
    # If in a group, add group stats
    if chat.type in ["group", "supergroup"]:
        chat_stats = db.get_chat_stats(chat.id)
        if chat_stats:
            stats_message += (
                f"\n📊 *Stats for {chat.title}* 📊\n\n"
                f"💬 Total messages: {chat_stats['message_count']}\n"
                f"🎭 Stickers sent: {chat_stats['sticker_count']}\n"
                f"📸 Photos shared: {chat_stats['photo_count']}\n"
                f"👥 Unique users: {chat_stats['unique_users']}\n"
            )
    
    # Create inline keyboard for visualizations
    keyboard = [
        [
            InlineKeyboardButton("Word Cloud", callback_data="stats_wordcloud"),
            InlineKeyboardButton("Mood Chart", callback_data="stats_mood")
        ]
    ]
    
    if db.list_memories(user.id):
        keyboard.append([InlineKeyboardButton("Memory Network", callback_data="stats_memory")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await message.reply_text(stats_message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

# === Stats Callback Handler ===
async def stats_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = query.from_user
    
    await query.answer()
    
    if query.data == "stats_wordcloud":
        await query.message.reply_chat_action(ChatAction.UPLOAD_PHOTO)
        wordcloud = await generate_word_cloud(user.id)
        
        if wordcloud:
            await query.message.reply_photo(wordcloud, caption="Your Word Cloud - Based on your messages")
        else:
            await query.message.reply_text("Not enough messages to generate a word cloud yet!")
            
    elif query.data == "stats_mood":
        await query.message.reply_chat_action(ChatAction.UPLOAD_PHOTO)
        mood_chart = await generate_mood_chart(user.id)
        
        if mood_chart:
            await query.message.reply_photo(mood_chart, caption="Your Mood History")
        else:
            await query.message.reply_text("Not enough mood data to generate a chart yet!")
            
    elif query.data == "stats_memory":
        await query.message.reply_chat_action(ChatAction.UPLOAD_PHOTO)
        memory_graph = await generate_memory_graph(user.id)
        
        if memory_graph:
            await query.message.reply_photo(memory_graph, caption="Your Memory Network")
        else:
            await query.message.reply_text("Not enough memories to generate a network yet!")

# === /settings Command ===
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user = message.from_user
    
    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    
    # Get current preferences
    prefs = db.get_user_preferences(user.id)
    
    # Create inline keyboard
    keyboard = [
        [
            InlineKeyboardButton("Language", callback_data="settings_lang"),
            InlineKeyboardButton(prefs['language'].capitalize(), callback_data="lang_toggle")
        ],
        [
            InlineKeyboardButton("Personality", callback_data="settings_personality"),
            InlineKeyboardButton(f"Level {prefs['personality_level']}", callback_data="personality_toggle")
        ],
        [
            InlineKeyboardButton("Notifications", callback_data="settings_notif"),
            InlineKeyboardButton("On" if prefs['notification_enabled'] else "Off", callback_data="notif_toggle")
        ],
        [
            InlineKeyboardButton("Auto Translate", callback_data="settings_translate"),
            InlineKeyboardButton("On" if prefs['auto_translate'] else "Off", callback_data="translate_toggle")
        ],
        [
            InlineKeyboardButton("News Category", callback_data="settings_news"),
            InlineKeyboardButton(prefs['preferred_news_category'].capitalize(), callback_data="news_category")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await message.reply_text(
        "⚙️ *Your Settings* ⚙️\n\n"
        "Customize how I interact with you:",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

# === Settings Callback Handler ===
async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = query.from_user
    
    # Get current preferences
    prefs = db.get_user_preferences(user.id)
    
    # Handle different callback actions
    if query.data == "lang_toggle":
        # Toggle language between hinglish and english
        new_lang = "english" if prefs['language'] == "hinglish" else "hinglish"
        db.set_user_preferences(user.id, language=new_lang)
        await query.answer(f"Language changed to {new_lang.capitalize()}")
    
    elif query.data == "personality_toggle":
        # Cycle through personality levels 1-3
        new_level = (prefs['personality_level'] % 3) + 1
        db.set_user_preferences(user.id, personality_level=new_level)
        await query.answer(f"Personality set to Level {new_level}")
    
    elif query.data == "notif_toggle":
        # Toggle notifications
        new_notif = not prefs['notification_enabled']
        db.set_user_preferences(user.id, notification_enabled=new_notif)
        await query.answer(f"Notifications turned {'on' if new_notif else 'off'}")
        
    elif query.data == "translate_toggle":
        # Toggle auto translate
        new_translate = not prefs['auto_translate']
        db.set_user_preferences(user.id, auto_translate=new_translate)
        await query.answer(f"Auto translate turned {'on' if new_translate else 'off'}")
        
    elif query.data == "news_category":
        # Show news category selection
        categories = ["general", "business", "entertainment", "health", "science", "sports", "technology"]
        
        # Create keyboard with categories
        keyboard = []
        for i in range(0, len(categories), 2):
            row = []
            for j in range(2):
                if i + j < len(categories):
                    category = categories[i + j]
                    row.append(InlineKeyboardButton(
                        f"{'✓ ' if category == prefs['preferred_news_category'] else ''}{category.capitalize()}", 
                        callback_data=f"news_set_{category}"
                    ))
            keyboard.append(row)
            
        # Add back button
        keyboard.append([InlineKeyboardButton("« Back to Settings", callback_data="back_to_settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "Select your preferred news category:",
            reply_markup=reply_markup
        )
        return
        
    elif query.data.startswith("news_set_"):
        # Set news category
        category = query.data.replace("news_set_", "")
        db.set_user_preferences(user.id, preferred_news_category=category)
        await query.answer(f"News category set to {category.capitalize()}")
        
    elif query.data == "back_to_settings":
        # Just continue to update the settings menu
        await query.answer()
    
    else:
        await query.answer()
        return
    
    # Update the keyboard with new settings
    prefs = db.get_user_preferences(user.id)  # Get updated preferences
    
    keyboard = [
        [
            InlineKeyboardButton("Language", callback_data="settings_lang"),
            InlineKeyboardButton(prefs['language'].capitalize(), callback_data="lang_toggle")
        ],
        [
            InlineKeyboardButton("Personality", callback_data="settings_personality"),
            InlineKeyboardButton(f"Level {prefs['personality_level']}", callback_data="personality_toggle")
        ],
        [
            InlineKeyboardButton("Notifications", callback_data="settings_notif"),
            InlineKeyboardButton("On" if prefs['notification_enabled'] else "Off", callback_data="notif_toggle")
        ],
        [
            InlineKeyboardButton("Auto Translate", callback_data="settings_translate"),
            InlineKeyboardButton("On" if prefs['auto_translate'] else "Off", callback_data="translate_toggle")
        ],
        [
            InlineKeyboardButton("News Category", callback_data="settings_news"),
            InlineKeyboardButton(prefs['preferred_news_category'].capitalize(), callback_data="news_category")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "⚙️ *Your Settings* ⚙️\n\n"
        "Customize how I interact with you:",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

# === /help Command ===
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    
    help_text = (
        "🌟 *Nikki Bot Commands* 🌟\n\n"
        "*Basic Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/settings - Customize bot settings\n\n"
        
        "*Fun Commands:*\n"
        "/name - Reply to a photo to identify anime characters\n"
        "/caption - Generate a caption for a photo\n"
        "/meme <text> - Create a meme with the replied photo\n\n"
        
        "*Utility Commands:*\n"
        "/weather <city> - Get weather information\n"
        "/news [category] - Get latest news\n"
        "/remind <time> <message> - Set a reminder\n"
        "/stats - View your usage statistics\n"
        "/translate <text> - Translate text to English\n"
        "/speak <text> - Convert text to speech\n\n"
        
        "*Memory Commands:*\n"
        "/remember <name> <info> - Store information about someone/something\n"
        "/recall <name> - Recall information about someone/something\n"
        "/forget <name> - Delete a memory\n"
        "/memories - List all your stored memories\n\n"
        
        "*How to use:*\n"
        "• In groups, reply to my messages or mention 'nikki' in your message\n"
        "• Send stickers as a reply to my messages for a reaction\n"
        "• Voice messages are automatically transcribed\n"
        "• For the best experience, try different personality levels in settings!\n\n"
        
        "Created with ❤️ by @sukuna_dev"
    )
    
    await message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# === /translate Command ===
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if text is provided
    if not args and not message.reply_to_message:
        await message.reply_text(
            "Please provide text to translate or reply to a message.\n"
            "Example: /translate Hello, how are you?"
        )
        return
    
    # Get text to translate
    if args:
        text_to_translate = " ".join(args)
    elif message.reply_to_message and message.reply_to_message.text:
        text_to_translate = message.reply_to_message.text
    else:
        await message.reply_text("I can only translate text messages.")
        return
    
    # Show typing indicator
    await message.reply_chat_action(ChatAction.TYPING)
    
    # Translate to English by default
    translated_text = await translate_text(text_to_translate)
    
    if translated_text == text_to_translate:
        await message.reply_text("Translation not available or text is already in English.")
    else:
        await message.reply_text(
            f"*Original:*\n{text_to_translate}\n\n"
            f"*Translation:*\n{translated_text}",
            parse_mode=ParseMode.MARKDOWN
        )

# === /speak Command ===
async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if text is provided
    if not args and not message.reply_to_message:
        await message.reply_text(
            "Please provide text to speak or reply to a message.\n"
            "Example: /speak Hello, how are you?"
        )
        return
    
    # Get text to speak
    if args:
        text_to_speak = " ".join(args)
    elif message.reply_to_message and message.reply_to_message.text:
        text_to_speak = message.reply_to_message.text
    else:
        await message.reply_text("I can only speak text messages.")
        return
    
    # Show typing indicator
    await message.reply_chat_action(ChatAction.RECORD_AUDIO)
    
    # Convert text to speech
    audio_file = await text_to_speech(text_to_speak)
    
    if audio_file:
        # Send audio file
        with open(audio_file, 'rb') as audio:
            await message.reply_voice(audio)
        
        # Clean up
        os.remove(audio_file)
    else:
        await message.reply_text("Sorry, I couldn't convert that text to speech.")

# === Memory Commands ===

# /remember command
async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if arguments are provided
    if len(args) < 2:
        await message.reply_text(
            "Format: /remember <name> <information>\n"
            "Example: /remember John John is my best friend who likes pizza\n\n"
            "You can also specify type and tags:\n"
            "/remember <name> type:<type> tags:<tag1,tag2> <information>"
        )
        return

# Continuing from previous file...

    # Parse arguments
    entity_name = args[0]
    
    # Check for type and tags
    entity_type = "person"  # Default type
    tags = []
    info_start_idx = 1
    
    for i, arg in enumerate(args[1:], 1):
        if arg.startswith("type:"):
            entity_type = arg[5:]
            info_start_idx = i + 1
        elif arg.startswith("tags:"):
            tags_str = arg[5:]
            tags = [tag.strip() for tag in tags_str.split(",")]
            info_start_idx = i + 1
    
    # Get the information
    if info_start_idx >= len(args):
        await message.reply_text("Please provide information about this entity.")
        return
        
    information = " ".join(args[info_start_idx:])
    
    # Store in database
    memory_id = db.add_memory(user.id, entity_name, entity_type, information, importance=1, tags=tags)
    
    await message.reply_text(
        f"✅ I'll remember that {entity_name} {information}\n"
        f"You can recall this anytime with /recall {entity_name}"
    )

# /recall command
async def recall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if name is provided
    if not args:
        # If no args, show all memories
        memories = db.list_memories(user.id, limit=10)
        if not memories:
            await message.reply_text("You haven't stored any memories yet.")
            return
            
        memory_text = "*Your Memories:*\n\n"
        for memory in memories:
            memory_text += f"• *{memory['entity_name']}* ({memory['entity_type']}): {memory['information'][:50]}...\n"
        
        memory_text += "\nUse /recall <name> to see full details."
        
        await message.reply_text(memory_text, parse_mode=ParseMode.MARKDOWN)
        return
    
    entity_name = args[0]
    
    # Get from database
    memory = db.get_memory(user.id, entity_name)
    
    if memory:
        # Format tags
        tags_text = ""
        if memory['tags']:
            tags_text = f"\n*Tags:* {', '.join(memory['tags'])}"
            
        # Get related memories
        related = db.get_related_memories(memory['id'], limit=3)
        related_text = ""
        
        if related:
            related_text = "\n\n*Related:*\n"
            for rel in related:
                related_text += f"• {rel['entity_name']} ({rel['relationship_type']})\n"
        
        await message.reply_text(
            f"*{memory['entity_name']}* ({memory['entity_type']})\n\n"
            f"{memory['information']}"
            f"{tags_text}"
            f"{related_text}",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        # Try searching
        search_results = db.search_memories(user.id, entity_name)
        
        if search_results:
            await message.reply_text(
                f"I don't remember anything specific about '{entity_name}', but I found these related memories:\n\n" +
                "\n".join([f"• *{m['entity_name']}*: {m['information'][:50]}..." for m in search_results]) +
                "\n\nUse /recall <name> to see details about any of these.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await message.reply_text(f"I don't remember anything about '{entity_name}'.")

# /forget command
async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if name is provided
    if not args:
        await message.reply_text("Please specify what to forget: /forget <name>")
        return
    
    entity_name = args[0]
    
    # Delete from database
    success = db.delete_memory(user.id, entity_name=entity_name)
    
    if success:
        await message.reply_text(f"I've forgotten everything about '{entity_name}'.")
    else:
        await message.reply_text(f"I don't have any memories about '{entity_name}'.")

# /memories command
async def memories_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat = message.chat
    user = message.from_user
    args = context.args

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    # Check if type filter is provided
    entity_type = None
    if args:
        entity_type = args[0]
    
    # Get memories from database
    memories = db.list_memories(user.id, entity_type=entity_type)
    
    if not memories:
        type_text = f" of type '{entity_type}'" if entity_type else ""
        await message.reply_text(f"You haven't stored any memories{type_text} yet.")
        return
    
    # Group memories by type
    memories_by_type = {}
    for memory in memories:
        if memory['entity_type'] not in memories_by_type:
            memories_by_type[memory['entity_type']] = []
        memories_by_type[memory['entity_type']].append(memory)
    
    # Format response
    memory_text = "*Your Memories:*\n\n"
    
    for type_name, type_memories in memories_by_type.items():
        memory_text += f"*{type_name.capitalize()}s:*\n"
        for memory in type_memories:
            # Truncate information if too long
            info = memory['information']
            if len(info) > 50:
                info = info[:47] + "..."
            
            memory_text += f"• *{memory['entity_name']}*: {info}\n"
        memory_text += "\n"
    
    memory_text += "Use /recall <name> to see full details about any memory."
    
    # Create inline keyboard for visualization
    keyboard = [[InlineKeyboardButton("View Memory Network", callback_data="stats_memory")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await message.reply_text(memory_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

# === Handle voice message ===
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    voice = message.voice
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'voice')

    # Check if this is a private chat or a reply to the bot in a group
    is_reply_to_nikki = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    )
    
    if chat.type in ["group", "supergroup"] and not is_reply_to_nikki:
        return

    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare voice messages? Thoda break le lo! 😅 1 minute baad try karo.")
        return

    user_name = user.full_name or user.username or "koi mystery cutie"
    user_id = user.id
    chat_id = chat.id
    chat_title = chat.title
    mention = user.mention_html()

    # Show typing indicator
    await message.reply_chat_action(ChatAction.TYPING)
    
    # Check if we already have a transcription for this voice message
    cached_transcription = db.get_voice_transcription(voice.file_id)
    
    if cached_transcription:
        transcription = cached_transcription
    else:
        # Transcribe voice message
        transcription = await transcribe_voice_message(voice)
        
        if not transcription:
            await message.reply_text("Sorry, I couldn't understand that voice message.")
            return
            
        # Cache the transcription
        db.save_voice_transcription(user_id, chat_id, voice.file_id, transcription)
    
    # Add transcription to message history
    message_history.add_message(user_id, chat_id, transcription)
    
    # Get relevant memories for this user
    memories = db.list_memories(user_id, limit=3)
    
    # Analyze sentiment
    mood, sentiment_score = analyze_sentiment(transcription)
    
    # Update user mood
    db.update_user_mood(user_id, mood, sentiment_score)
    
    # Generate response
    prompt = nikki_prompt(
        user_input=transcription,
        user_name=user_name,
        user_id=user_id,
        chat_id=chat_id,
        chat_title=chat_title,
        mention=mention,
        user_memories=memories
    )
    
    reply = generate_with_retry(prompt)
    
    # First confirm what was heard
    await message.reply_text(f"🎤 I heard: \"{transcription}\"")
    
    # Then respond
    await message.reply_html(reply)

# === Handle message ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    
    # Skip if no text
    if not message.text:
        return
        
    user_input = message.text
    chat = message.chat
    user = message.from_user

    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    db.update_chat_stats(chat.id, chat.title, 'text')

    user_name = user.full_name or user.username or "koi mystery cutie"
    user_id = user.id
    chat_id = chat.id
    chat_title = chat.title
    mention = user.mention_html()

    # Add user message to history
    message_history.add_message(user_id, chat_id, message.text)
    
    # Analyze sentiment
    mood, sentiment_score = analyze_sentiment(user_input)
    
    # Update user mood
    db.update_user_mood(user_id, mood, sentiment_score)

    if user_input.lower() in GREETING_RESPONSES:
        await message.reply_text(GREETING_RESPONSES[user_input.lower()])
        return

    is_reply_to_nikki = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    )

    # In groups, only respond to mentions, replies, or when "nikki" is in the message
    if chat.type in ["group", "supergroup"]:
        should_respond = False
        
        if is_reply_to_nikki:
            should_respond = True
        elif "nikki" in user_input.lower():
            should_respond = True
        elif f"@{context.bot.username}" in user_input.lower():
            should_respond = True
        
        if not should_respond:
            return

        if user_input.strip().lower() == "nikki":
            await message.reply_text("Hmm? Kisi ne mujhe bulaya kya?")
            return
        elif "are you a bot" in user_input.lower() or "you're a bot" in user_input.lower():
            await message.reply_text("Bot? Tujhe main unreal lagti hoon kya?")
            return
    
    # Check rate limit
    if not check_rate_limit(user.id):
        await message.reply_text("Arrey itne saare messages? Thoda break le lo! 😅 1 minute baad try karo.")
        return
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Get user preferences
    user_prefs = db.get_user_preferences(user.id)
    personality_level = user_prefs['personality_level']
    
    # Check if auto-translate is enabled
    if user_prefs['auto_translate'] and not any(word in user_input.lower() for word in ['nikki', 'hello', 'hi', 'hey']):
        # Detect if text is not in English or Hinglish
        if not re.search(r'[a-zA-Z]', user_input) or re.search(r'[\u0900-\u097F]', user_input):
            try:
                translated = await translate_text(user_input)
                if translated != user_input:
                    user_input = translated
                    await message.reply_text(f"(Translated: {translated})")
            except Exception as e:
                logging.error(f"Auto-translation error: {e}")
    
    # Get relevant memories for this user
    memories = db.list_memories(user_id, limit=3)
    
    # Check for memory-related keywords
    memory_keywords = ["remember", "yaad", "recall", "tell me about", "who is", "what is", "where is", "when is"]
    
    if any(keyword in user_input.lower() for keyword in memory_keywords) and not user_input.startswith('/'):
        # Try to extract entity name
        entity_match = re.search(r'(?:about|who is|what is|where is|when is|yaad)\s+([a-zA-Z0-9_\s]+)', user_input.lower())
        
        if entity_match:
            entity_name = entity_match.group(1).strip()
            memory = db.get_memory(user_id, entity_name)
            
            if memory:
                # Include memory in prompt
                prompt = nikki_prompt(
                    user_input=user_input,
                    user_name=user_name,
                    user_id=user_id,
                    chat_id=chat_id,
                    chat_title=chat_title,
                    mention=mention,
                    personality_level=personality_level,
                    user_memories=[memory]
                )
                
                reply = generate_with_retry(prompt)
                await message.reply_html(reply)
                return
    
    prompt = nikki_prompt(
        user_input=user_input,
        user_name=user_name,
        user_id=user_id,
        chat_id=chat_id,
        chat_title=chat_title,
        mention=mention,
        personality_level=personality_level,
        user_memories=memories
    )
    
    if prompt is None:
        return
        
    reply = generate_with_retry(prompt)
    await message.reply_html(reply)

# === Check for due reminders ===
async def check_reminders(context: CallbackContext):
    try:
        reminders = db.get_due_reminders()
        
        for reminder in reminders:
            reminder_id, user_id, chat_id, reminder_text, is_recurring, recurrence_pattern = reminder
            
            try:
                recurring_text = f" ({recurrence_pattern})" if is_recurring else ""
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"⏰ *Reminder{recurring_text}* ⏰\n\n{reminder_text}",
                    parse_mode=ParseMode.MARKDOWN
                )
                logging.info(f"Sent reminder {reminder_id} to user {user_id} in chat {chat_id}")
            except Exception as e:
                logging.error(f"Failed to send reminder {reminder_id}: {e}")
    except Exception as e:
        logging.error(f"Error checking reminders: {e}")

# === Start command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    
    # Update database
    db.update_user_profile(user.id, user.username, user.full_name)
    
    welcome_text = (
        f"Hii {user.first_name}! 😍 Nikki yaha hai...\n\n"
        "Main aapki kya help kar sakti hoon?\n"
        "• Chat kar sakte ho mujhse kisi bhi topic pe\n"
        "• Stickers bhej sakte ho, main samajh lungi\n"
        "• Voice messages bhi support karti hoon\n"
        "• Photos pe /name, /caption, ya /meme commands try karo\n"
        "• /remember command se mujhe kuch bhi yaad karwa sakte ho\n"
        "• /weather, /news, /remind jaise useful commands bhi hain\n\n"
        "Settings customize karne ke liye /settings use karo\n"
        "Saare commands dekhne ke liye /help type karo\n\n"
        "Let's have some fun! 💕"
    )
    
    # Create welcome keyboard
    keyboard = [
        [
            InlineKeyboardButton("🛠️ Settings", callback_data="cmd_settings"),
            InlineKeyboardButton("❓ Help", callback_data="cmd_help")
        ],
        [
            InlineKeyboardButton("🧠 Memory", callback_data="cmd_memory"),
            InlineKeyboardButton("📊 Stats", callback_data="cmd_stats")
        ],
        [
            InlineKeyboardButton("🌤️ Weather", callback_data="cmd_weather"),
            InlineKeyboardButton("📰 News", callback_data="cmd_news")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def welcome_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "cmd_settings":
        await settings_command(update, context)
    elif query.data == "cmd_help":
        await help_command(update, context)
    elif query.data == "cmd_memory":
        await memories_command(update, context)
    elif query.data == "cmd_stats":
        await stats_command(update, context)
    elif query.data == "cmd_weather":
        await query.edit_message_text("Use /weather <city> to get weather information!")
    elif query.data == "cmd_news":
        await query.edit_message_text("Use /news [category] to get the latest news!")

async def rollout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chalo chalo, Nikki aa gayi... ab toh mood hi ban gaya!")

async def clear_old_updates(app):
    try:
        updates = await app.bot.get_updates(timeout=10)
        if updates:
            latest_update_id = max(update.update_id for update in updates)
            await app.bot.get_updates(offset=latest_update_id + 1, timeout=10)
            logging.info(f"Cleared {len(updates)} old updates.")
        else:
            logging.info("No old updates to clear.")
    except Exception as e:
        logging.error(f"Error clearing old updates: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Exception while handling an update: {context.error}")
    
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = ''.join(tb_list)
    logging.error(f"Traceback:\n{tb_string}")
    if update and update.effective_message and (
        update.effective_message.text.startswith('/') or 
        update.effective_message.chat.type == 'private'
    ):
        await update.effective_message.reply_text(
            "Oops! Something went wrong while processing your request. Please try again later."
        )


BOT_NAME = "Nikki_ChatBot_Bot"  # Replace with your bot's name
CREATOR = "@ZyroAss"  # Your tag
IMAGE_URL = "https://files.catbox.moe/94pfe0.jpg"  # Replace with your bot's photo URL

START_CAPTION = f"""╭───────────────────⦿
│❖ 𝖧ᴇʏ ɪ'ᴍ {BOT_NAME}
├───────────────────⦿
│✦ ɪ ʜᴀᴠᴇ ᴍᴀɢɪᴄ ғᴇᴀᴛᴜʀᴇs.
│❍ 𝖠ᴅᴠᴀɴᴄᴇᴅ ᴀɪ ʙᴀsᴇᴅ ᴄʜᴀᴛʙᴏᴛ.
├───────────────────⦿
│✦ ɪ'ᴍ ꜱᴍᴀʀᴛ & ᴀʙᴜꜱᴇʟᴇꜱꜱ ᴄʜᴀᴛʙᴏᴛ.
│❍ ɪ ᴄᴀɴ ʜᴇʟᴘ ᴀᴄᴛɪᴠᴇ ʏᴏᴜʀ ɢʀᴏᴜᴘ.
│✦ I ᴄᴀɴ ᴛᴇʟʟ ʏᴏᴜ ᴛɪᴍᴇ ɪꜰ ʏᴏᴜ ᴀꜱᴋ.
├───────────────────⦿
│❖ ϻᴀᴅᴇ ʙʏ {CREATOR}
╰───────────────────⦿"""

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("➕ Add to Your Group", url=f"https://t.me/{context.bot.username}?startgroup=true")],
        [
            InlineKeyboardButton("💬 Support", url="https://t.me/Zyroupdates"),
            InlineKeyboardButton("📢 Updates", url="https://t.me/ZyroBotCodes")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=IMAGE_URL,
        caption=START_CAPTION,
        reply_markup=reply_markup
    )

# Run bot



from telegram.ext import ChatMemberHandler

async def my_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_status = update.my_chat_member.new_chat_member.status
    old_status = update.my_chat_member.old_chat_member.status
    chat = update.my_chat_member.chat
    user = update.my_chat_member.from_user

    if new_status == "member":
        # Bot added to a group
        try:
            member_count = await context.bot.get_chat_member_count(chat.id)
        except:
            member_count = "unknown"
        text = f"👋 *Added to New Group!*\n\n" \
               f"👤 Added by: [{user.full_name}](tg://user?id={user.id})\n" \
               f"📛 Group: {chat.title}\n" \
               f"👥 Members: {member_count}"
        await context.bot.send_message(chat_id=MAIN_GC_ID, text=text, parse_mode=ParseMode.MARKDOWN)

    elif new_status in ["left", "kicked"]:
        text = f"👋 *Removed from Group*\n\n" \
               f"📛 Group: {chat.title}\n" \
               f"❌ By: [{user.full_name}](tg://user?id={user.id})"
        await context.bot.send_message(chat_id=MAIN_GC_ID, text=text, parse_mode=ParseMode.MARKDOWN)


if __name__ == "__main__":
    setup_database()
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("FFmpeg is installed and working.")
    except Exception as e:
        logging.error(f"FFmpeg is not installed or not working: {e}")
        print("ERROR: FFmpeg is required for video sticker processing. Please install it.")
        print("On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("On CentOS/RHEL: sudo yum install ffmpeg")
        print("On macOS: brew install ffmpeg")
        print("On Windows: Download from https://ffmpeg.org/download.html")
        exit(1)
    temp_dir = "temp_thumbnails"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        logging.info(f"Created temporary directory: {temp_dir}")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("rollout", rollout))
    app.add_handler(CommandHandler("name", name_command))
    app.add_handler(CommandHandler("caption", caption_command))
    app.add_handler(CommandHandler("meme", meme_command))
    app.add_handler(CommandHandler("weather", weather_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("remind", remind_command))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("speak", speak_command))
    app.add_handler(CommandHandler("remember", remember_command))
    app.add_handler(CommandHandler("recall", recall_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("memories", memories_command))
    app.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^(lang_toggle|personality_toggle|notif_toggle|translate_toggle|news_category|news_set_|back_to_settings)"))
    app.add_handler(CallbackQueryHandler(welcome_callback, pattern=r"^cmd_"))
    app.add_handler(CallbackQueryHandler(stats_callback, pattern=r"^stats_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
    #app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    #app.add_handler(ChatMemberHandler(my_chat_member_handler, chat_member_types=ChatMemberHandler.MY_CHAT_MEMBER))

    app.add_error_handler(error_handler)

    job_queue = app.job_queue

    job_queue.run_repeating(check_reminders, interval=60, first=10)
    
    print("Nikki is online. Super advanced version with memory system and many new features is ready!")
    app.run_polling()
