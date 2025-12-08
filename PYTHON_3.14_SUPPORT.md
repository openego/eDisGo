# Python 3.14 Support

This branch adds Python 3.14 support to eDisGo.

## Changes Made

1. **Updated eDisGo_env_dev.yml**
   - Changed Python version constraint from `<= 3.13` to `<= 3.14`

2. **Updated CI/CD workflow**
   - Added Python 3.14 to the test matrix in `.github/workflows/tests-coverage.yml`

3. **Inherited from Python 3.13 support**
   - Removed `pygeos` dependency (incompatible with Python 3.12+)
   - Updated numpy constraint to allow version 2.x for newer Python versions

## Known Issues

### NumPy Version
- NumPy 1.26.4 only supports up to Python 3.12
- NumPy 2.1.0+ is required for Python 3.13 and 3.14
- The constraint has been updated to `numpy >= 1.26.4` to allow both versions

### PyGEOS Removal
- PyGEOS is incompatible with Python 3.12+ due to removal of `configparser.SafeConfigParser`
- PyGEOS functionality is now integrated into Shapely 2.0+
- GeoPandas 0.12+ uses Shapely 2.0, so PyGEOS is no longer needed

## Testing

Python 3.14 is currently in alpha/early development stage. Testing will be possible when:
- GitHub Actions adds Python 3.14 support
- Python 3.14 is officially released

The CI/CD pipeline will automatically test against Python 3.14 once it becomes available in GitHub Actions.

## Next Steps

1. Monitor CI/CD runs once Python 3.14 support is available in GitHub Actions
2. Fix any compatibility issues that emerge during testing
3. Update dependencies as needed for Python 3.14 compatibility
