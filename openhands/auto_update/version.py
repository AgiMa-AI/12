"""
Version definitions for auto-update.

This module provides version definitions for auto-update,
including version information and comparison.
"""

import re
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class Version:
    """Version class for semantic versioning."""
    
    def __init__(self, major: int, minor: int, patch: int, pre_release: Optional[str] = None, build: Optional[str] = None):
        """
        Initialize version.
        
        Args:
            major: Major version
            minor: Minor version
            patch: Patch version
            pre_release: Pre-release identifier
            build: Build metadata
        """
        self.major = major
        self.minor = minor
        self.patch = patch
        self.pre_release = pre_release
        self.build = build
    
    def __str__(self) -> str:
        """
        Convert to string.
        
        Returns:
            String representation
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.pre_release:
            version += f"-{self.pre_release}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __eq__(self, other: object) -> bool:
        """
        Check if versions are equal.
        
        Args:
            other: Other version
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, Version):
            return NotImplemented
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.pre_release == other.pre_release
        )
    
    def __lt__(self, other: "Version") -> bool:
        """
        Check if version is less than other version.
        
        Args:
            other: Other version
            
        Returns:
            True if less than, False otherwise
        """
        if self.major != other.major:
            return self.major < other.major
        
        if self.minor != other.minor:
            return self.minor < other.minor
        
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # Pre-release versions are less than the associated normal version
        if self.pre_release is None and other.pre_release is not None:
            return False
        
        if self.pre_release is not None and other.pre_release is None:
            return True
        
        # Compare pre-release identifiers
        if self.pre_release != other.pre_release:
            # Split pre-release identifiers into parts
            self_parts = self.pre_release.split(".")
            other_parts = other.pre_release.split(".")
            
            # Compare parts
            for i in range(min(len(self_parts), len(other_parts))):
                self_part = self_parts[i]
                other_part = other_parts[i]
                
                # Numeric parts are compared numerically
                if self_part.isdigit() and other_part.isdigit():
                    if int(self_part) != int(other_part):
                        return int(self_part) < int(other_part)
                
                # Numeric parts are less than non-numeric parts
                elif self_part.isdigit() and not other_part.isdigit():
                    return True
                
                elif not self_part.isdigit() and other_part.isdigit():
                    return False
                
                # Non-numeric parts are compared lexically
                elif self_part != other_part:
                    return self_part < other_part
            
            # If all parts are equal, the one with fewer parts is less
            return len(self_parts) < len(other_parts)
        
        return False
    
    def __le__(self, other: "Version") -> bool:
        """
        Check if version is less than or equal to other version.
        
        Args:
            other: Other version
            
        Returns:
            True if less than or equal, False otherwise
        """
        return self < other or self == other
    
    def __gt__(self, other: "Version") -> bool:
        """
        Check if version is greater than other version.
        
        Args:
            other: Other version
            
        Returns:
            True if greater than, False otherwise
        """
        return not (self <= other)
    
    def __ge__(self, other: "Version") -> bool:
        """
        Check if version is greater than or equal to other version.
        
        Args:
            other: Other version
            
        Returns:
            True if greater than or equal, False otherwise
        """
        return not (self < other)
    
    @classmethod
    def parse(cls, version_string: str) -> "Version":
        """
        Parse version string.
        
        Args:
            version_string: Version string
            
        Returns:
            Version
            
        Raises:
            ValueError: If version string is invalid
        """
        # Regular expression for semantic versioning
        pattern = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        
        match = re.match(pattern, version_string)
        if not match:
            raise ValueError(f"Invalid version string: {version_string}")
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        pre_release = match.group(4)
        build = match.group(5)
        
        return cls(major, minor, patch, pre_release, build)

class VersionInfo:
    """Version information."""
    
    def __init__(self, 
                version: Version,
                release_date: Optional[datetime.datetime] = None,
                changelog: Optional[str] = None,
                download_url: Optional[str] = None,
                size: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize version information.
        
        Args:
            version: Version
            release_date: Release date
            changelog: Changelog
            download_url: Download URL
            size: Size in bytes
            metadata: Additional metadata
        """
        self.version = version
        self.release_date = release_date or datetime.datetime.now()
        self.changelog = changelog or ""
        self.download_url = download_url or ""
        self.size = size or 0
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "version": str(self.version),
            "release_date": self.release_date.isoformat(),
            "changelog": self.changelog,
            "download_url": self.download_url,
            "size": self.size,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionInfo":
        """
        Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Version information
        """
        version = Version.parse(data["version"])
        
        release_date = None
        if "release_date" in data:
            try:
                release_date = datetime.datetime.fromisoformat(data["release_date"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            version=version,
            release_date=release_date,
            changelog=data.get("changelog"),
            download_url=data.get("download_url"),
            size=data.get("size"),
            metadata=data.get("metadata", {})
        )