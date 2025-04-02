"""
Repository definitions for auto-update.

This module provides repository definitions for auto-update,
including repository types and repository classes.
"""

import os
import json
import enum
import logging
import datetime
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional, Union, Tuple

from openhands.auto_update.version import Version, VersionInfo

logger = logging.getLogger("auto_update_repository")

class RepositoryType(enum.Enum):
    """Repository type enumeration."""
    LOCAL = "local"
    HTTP = "http"
    GITHUB = "github"
    CUSTOM = "custom"

class Repository:
    """Repository class for auto-update."""
    
    def __init__(self, 
                name: str,
                repository_type: RepositoryType,
                url: str,
                username: Optional[str] = None,
                password: Optional[str] = None,
                token: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize repository.
        
        Args:
            name: Repository name
            repository_type: Repository type
            url: Repository URL
            username: Username for authentication
            password: Password for authentication
            token: Token for authentication
            metadata: Additional metadata
        """
        self.name = name
        self.repository_type = repository_type
        self.url = url
        self.username = username
        self.password = password
        self.token = token
        self.metadata = metadata or {}
    
    def get_latest_version(self, component: str) -> Optional[VersionInfo]:
        """
        Get latest version information.
        
        Args:
            component: Component name
            
        Returns:
            Latest version information or None if not found
        """
        if self.repository_type == RepositoryType.LOCAL:
            return self._get_latest_version_local(component)
        elif self.repository_type == RepositoryType.HTTP:
            return self._get_latest_version_http(component)
        elif self.repository_type == RepositoryType.GITHUB:
            return self._get_latest_version_github(component)
        else:
            logger.warning(f"Unsupported repository type: {self.repository_type.value}")
            return None
    
    def get_version(self, component: str, version: Version) -> Optional[VersionInfo]:
        """
        Get specific version information.
        
        Args:
            component: Component name
            version: Version
            
        Returns:
            Version information or None if not found
        """
        if self.repository_type == RepositoryType.LOCAL:
            return self._get_version_local(component, version)
        elif self.repository_type == RepositoryType.HTTP:
            return self._get_version_http(component, version)
        elif self.repository_type == RepositoryType.GITHUB:
            return self._get_version_github(component, version)
        else:
            logger.warning(f"Unsupported repository type: {self.repository_type.value}")
            return None
    
    def get_versions(self, component: str) -> List[VersionInfo]:
        """
        Get all version information.
        
        Args:
            component: Component name
            
        Returns:
            List of version information
        """
        if self.repository_type == RepositoryType.LOCAL:
            return self._get_versions_local(component)
        elif self.repository_type == RepositoryType.HTTP:
            return self._get_versions_http(component)
        elif self.repository_type == RepositoryType.GITHUB:
            return self._get_versions_github(component)
        else:
            logger.warning(f"Unsupported repository type: {self.repository_type.value}")
            return []
    
    def download_version(self, component: str, version: Version, destination: str) -> bool:
        """
        Download specific version.
        
        Args:
            component: Component name
            version: Version
            destination: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        if self.repository_type == RepositoryType.LOCAL:
            return self._download_version_local(component, version, destination)
        elif self.repository_type == RepositoryType.HTTP:
            return self._download_version_http(component, version, destination)
        elif self.repository_type == RepositoryType.GITHUB:
            return self._download_version_github(component, version, destination)
        else:
            logger.warning(f"Unsupported repository type: {self.repository_type.value}")
            return False
    
    def _get_latest_version_local(self, component: str) -> Optional[VersionInfo]:
        """
        Get latest version information from local repository.
        
        Args:
            component: Component name
            
        Returns:
            Latest version information or None if not found
        """
        try:
            # Get component directory
            component_dir = os.path.join(self.url, component)
            
            # Check if component directory exists
            if not os.path.exists(component_dir) or not os.path.isdir(component_dir):
                logger.warning(f"Component directory not found: {component_dir}")
                return None
            
            # Get versions
            versions = []
            for version_dir in os.listdir(component_dir):
                try:
                    version = Version.parse(version_dir)
                    versions.append(version)
                except ValueError:
                    pass
            
            if not versions:
                logger.warning(f"No versions found for component: {component}")
                return None
            
            # Get latest version
            latest_version = max(versions)
            
            # Get version information
            return self._get_version_local(component, latest_version)
        
        except Exception as e:
            logger.error(f"Failed to get latest version for component {component}: {e}")
            return None
    
    def _get_version_local(self, component: str, version: Version) -> Optional[VersionInfo]:
        """
        Get specific version information from local repository.
        
        Args:
            component: Component name
            version: Version
            
        Returns:
            Version information or None if not found
        """
        try:
            # Get version directory
            version_dir = os.path.join(self.url, component, str(version))
            
            # Check if version directory exists
            if not os.path.exists(version_dir) or not os.path.isdir(version_dir):
                logger.warning(f"Version directory not found: {version_dir}")
                return None
            
            # Get version information file
            version_info_file = os.path.join(version_dir, "version.json")
            
            # Check if version information file exists
            if not os.path.exists(version_info_file):
                logger.warning(f"Version information file not found: {version_info_file}")
                
                # Create basic version information
                return VersionInfo(
                    version=version,
                    release_date=datetime.datetime.fromtimestamp(os.path.getctime(version_dir)),
                    download_url=version_dir
                )
            
            # Load version information
            with open(version_info_file, "r", encoding="utf-8") as f:
                version_info_data = json.load(f)
            
            # Create version information
            return VersionInfo.from_dict(version_info_data)
        
        except Exception as e:
            logger.error(f"Failed to get version {version} for component {component}: {e}")
            return None
    
    def _get_versions_local(self, component: str) -> List[VersionInfo]:
        """
        Get all version information from local repository.
        
        Args:
            component: Component name
            
        Returns:
            List of version information
        """
        try:
            # Get component directory
            component_dir = os.path.join(self.url, component)
            
            # Check if component directory exists
            if not os.path.exists(component_dir) or not os.path.isdir(component_dir):
                logger.warning(f"Component directory not found: {component_dir}")
                return []
            
            # Get versions
            versions = []
            for version_dir in os.listdir(component_dir):
                try:
                    version = Version.parse(version_dir)
                    version_info = self._get_version_local(component, version)
                    if version_info:
                        versions.append(version_info)
                except ValueError:
                    pass
            
            # Sort versions
            versions.sort(key=lambda x: x.version)
            
            return versions
        
        except Exception as e:
            logger.error(f"Failed to get versions for component {component}: {e}")
            return []
    
    def _download_version_local(self, component: str, version: Version, destination: str) -> bool:
        """
        Download specific version from local repository.
        
        Args:
            component: Component name
            version: Version
            destination: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            
            # Get version directory
            version_dir = os.path.join(self.url, component, str(version))
            
            # Check if version directory exists
            if not os.path.exists(version_dir) or not os.path.isdir(version_dir):
                logger.warning(f"Version directory not found: {version_dir}")
                return False
            
            # Create destination directory if it doesn't exist
            os.makedirs(destination, exist_ok=True)
            
            # Copy files
            for item in os.listdir(version_dir):
                if item == "version.json":
                    continue
                
                source_item = os.path.join(version_dir, item)
                dest_item = os.path.join(destination, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
            
            logger.info(f"Downloaded version {version} for component {component} to {destination}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download version {version} for component {component}: {e}")
            return False
    
    def _get_latest_version_http(self, component: str) -> Optional[VersionInfo]:
        """
        Get latest version information from HTTP repository.
        
        Args:
            component: Component name
            
        Returns:
            Latest version information or None if not found
        """
        try:
            # Get versions
            versions = self._get_versions_http(component)
            
            if not versions:
                logger.warning(f"No versions found for component: {component}")
                return None
            
            # Get latest version
            latest_version = max(versions, key=lambda x: x.version)
            
            return latest_version
        
        except Exception as e:
            logger.error(f"Failed to get latest version for component {component}: {e}")
            return None
    
    def _get_version_http(self, component: str, version: Version) -> Optional[VersionInfo]:
        """
        Get specific version information from HTTP repository.
        
        Args:
            component: Component name
            version: Version
            
        Returns:
            Version information or None if not found
        """
        try:
            # Get version information URL
            version_info_url = f"{self.url}/{component}/{version}/version.json"
            
            # Create request
            request = urllib.request.Request(version_info_url)
            
            # Add authentication if provided
            if self.username and self.password:
                import base64
                auth = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
                request.add_header("Authorization", f"Basic {auth}")
            elif self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Get version information
            with urllib.request.urlopen(request) as response:
                version_info_data = json.loads(response.read().decode())
            
            # Create version information
            return VersionInfo.from_dict(version_info_data)
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Version {version} not found for component {component}")
            else:
                logger.error(f"Failed to get version {version} for component {component}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to get version {version} for component {component}: {e}")
            return None
    
    def _get_versions_http(self, component: str) -> List[VersionInfo]:
        """
        Get all version information from HTTP repository.
        
        Args:
            component: Component name
            
        Returns:
            List of version information
        """
        try:
            # Get versions URL
            versions_url = f"{self.url}/{component}/versions.json"
            
            # Create request
            request = urllib.request.Request(versions_url)
            
            # Add authentication if provided
            if self.username and self.password:
                import base64
                auth = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
                request.add_header("Authorization", f"Basic {auth}")
            elif self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Get versions
            with urllib.request.urlopen(request) as response:
                versions_data = json.loads(response.read().decode())
            
            # Create version information
            versions = []
            for version_data in versions_data:
                try:
                    version_info = VersionInfo.from_dict(version_data)
                    versions.append(version_info)
                except Exception as e:
                    logger.error(f"Failed to parse version information: {e}")
            
            # Sort versions
            versions.sort(key=lambda x: x.version)
            
            return versions
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Versions not found for component {component}")
            else:
                logger.error(f"Failed to get versions for component {component}: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Failed to get versions for component {component}: {e}")
            return []
    
    def _download_version_http(self, component: str, version: Version, destination: str) -> bool:
        """
        Download specific version from HTTP repository.
        
        Args:
            component: Component name
            version: Version
            destination: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get version information
            version_info = self._get_version_http(component, version)
            if not version_info:
                logger.warning(f"Version {version} not found for component {component}")
                return False
            
            # Get download URL
            download_url = version_info.download_url
            if not download_url:
                logger.warning(f"Download URL not found for version {version} of component {component}")
                return False
            
            # Create destination directory if it doesn't exist
            os.makedirs(destination, exist_ok=True)
            
            # Download file
            dest_file = os.path.join(destination, os.path.basename(download_url))
            
            # Create request
            request = urllib.request.Request(download_url)
            
            # Add authentication if provided
            if self.username and self.password:
                import base64
                auth = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
                request.add_header("Authorization", f"Basic {auth}")
            elif self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Download file
            with urllib.request.urlopen(request) as response, open(dest_file, "wb") as f:
                f.write(response.read())
            
            # Extract file if it's an archive
            if dest_file.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(dest_file, "r") as zip_ref:
                    zip_ref.extractall(destination)
                os.remove(dest_file)
            elif dest_file.endswith(".tar.gz") or dest_file.endswith(".tgz"):
                import tarfile
                with tarfile.open(dest_file, "r:gz") as tar_ref:
                    tar_ref.extractall(destination)
                os.remove(dest_file)
            
            logger.info(f"Downloaded version {version} for component {component} to {destination}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download version {version} for component {component}: {e}")
            return False
    
    def _get_latest_version_github(self, component: str) -> Optional[VersionInfo]:
        """
        Get latest version information from GitHub repository.
        
        Args:
            component: Component name
            
        Returns:
            Latest version information or None if not found
        """
        try:
            # Get latest release URL
            latest_release_url = f"https://api.github.com/repos/{self.url}/{component}/releases/latest"
            
            # Create request
            request = urllib.request.Request(latest_release_url)
            
            # Add authentication if provided
            if self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Get latest release
            with urllib.request.urlopen(request) as response:
                release_data = json.loads(response.read().decode())
            
            # Parse version
            version_string = release_data["tag_name"]
            if version_string.startswith("v"):
                version_string = version_string[1:]
            
            version = Version.parse(version_string)
            
            # Create version information
            return VersionInfo(
                version=version,
                release_date=datetime.datetime.fromisoformat(release_data["published_at"].replace("Z", "+00:00")),
                changelog=release_data["body"],
                download_url=release_data["zipball_url"],
                size=0,
                metadata={
                    "html_url": release_data["html_url"],
                    "assets": [
                        {
                            "name": asset["name"],
                            "url": asset["browser_download_url"],
                            "size": asset["size"]
                        }
                        for asset in release_data["assets"]
                    ]
                }
            )
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Latest release not found for component {component}")
            else:
                logger.error(f"Failed to get latest release for component {component}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to get latest release for component {component}: {e}")
            return None
    
    def _get_version_github(self, component: str, version: Version) -> Optional[VersionInfo]:
        """
        Get specific version information from GitHub repository.
        
        Args:
            component: Component name
            version: Version
            
        Returns:
            Version information or None if not found
        """
        try:
            # Get release URL
            release_url = f"https://api.github.com/repos/{self.url}/{component}/releases/tags/v{version}"
            
            # Create request
            request = urllib.request.Request(release_url)
            
            # Add authentication if provided
            if self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Get release
            with urllib.request.urlopen(request) as response:
                release_data = json.loads(response.read().decode())
            
            # Create version information
            return VersionInfo(
                version=version,
                release_date=datetime.datetime.fromisoformat(release_data["published_at"].replace("Z", "+00:00")),
                changelog=release_data["body"],
                download_url=release_data["zipball_url"],
                size=0,
                metadata={
                    "html_url": release_data["html_url"],
                    "assets": [
                        {
                            "name": asset["name"],
                            "url": asset["browser_download_url"],
                            "size": asset["size"]
                        }
                        for asset in release_data["assets"]
                    ]
                }
            )
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Release {version} not found for component {component}")
            else:
                logger.error(f"Failed to get release {version} for component {component}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to get release {version} for component {component}: {e}")
            return None
    
    def _get_versions_github(self, component: str) -> List[VersionInfo]:
        """
        Get all version information from GitHub repository.
        
        Args:
            component: Component name
            
        Returns:
            List of version information
        """
        try:
            # Get releases URL
            releases_url = f"https://api.github.com/repos/{self.url}/{component}/releases"
            
            # Create request
            request = urllib.request.Request(releases_url)
            
            # Add authentication if provided
            if self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Get releases
            with urllib.request.urlopen(request) as response:
                releases_data = json.loads(response.read().decode())
            
            # Create version information
            versions = []
            for release_data in releases_data:
                try:
                    # Parse version
                    version_string = release_data["tag_name"]
                    if version_string.startswith("v"):
                        version_string = version_string[1:]
                    
                    version = Version.parse(version_string)
                    
                    # Create version information
                    version_info = VersionInfo(
                        version=version,
                        release_date=datetime.datetime.fromisoformat(release_data["published_at"].replace("Z", "+00:00")),
                        changelog=release_data["body"],
                        download_url=release_data["zipball_url"],
                        size=0,
                        metadata={
                            "html_url": release_data["html_url"],
                            "assets": [
                                {
                                    "name": asset["name"],
                                    "url": asset["browser_download_url"],
                                    "size": asset["size"]
                                }
                                for asset in release_data["assets"]
                            ]
                        }
                    )
                    
                    versions.append(version_info)
                
                except Exception as e:
                    logger.error(f"Failed to parse release information: {e}")
            
            # Sort versions
            versions.sort(key=lambda x: x.version)
            
            return versions
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning(f"Releases not found for component {component}")
            else:
                logger.error(f"Failed to get releases for component {component}: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Failed to get releases for component {component}: {e}")
            return []
    
    def _download_version_github(self, component: str, version: Version, destination: str) -> bool:
        """
        Download specific version from GitHub repository.
        
        Args:
            component: Component name
            version: Version
            destination: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get version information
            version_info = self._get_version_github(component, version)
            if not version_info:
                logger.warning(f"Version {version} not found for component {component}")
                return False
            
            # Get download URL
            download_url = version_info.download_url
            if not download_url:
                logger.warning(f"Download URL not found for version {version} of component {component}")
                return False
            
            # Create destination directory if it doesn't exist
            os.makedirs(destination, exist_ok=True)
            
            # Download file
            dest_file = os.path.join(destination, f"{component}-{version}.zip")
            
            # Create request
            request = urllib.request.Request(download_url)
            
            # Add authentication if provided
            if self.token:
                request.add_header("Authorization", f"Bearer {self.token}")
            
            # Download file
            with urllib.request.urlopen(request) as response, open(dest_file, "wb") as f:
                f.write(response.read())
            
            # Extract file
            import zipfile
            with zipfile.ZipFile(dest_file, "r") as zip_ref:
                zip_ref.extractall(destination)
            
            # Remove zip file
            os.remove(dest_file)
            
            logger.info(f"Downloaded version {version} for component {component} to {destination}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download version {version} for component {component}: {e}")
            return False