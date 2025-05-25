import 'package:shared_preferences/shared_preferences.dart';

class SharedPreferencesService {
  SharedPreferencesService._privateConstructor();

  static final SharedPreferencesService instance =
      SharedPreferencesService._privateConstructor();

  // This Future will complete once, and then _prefs will hold the instance.
  Future<SharedPreferences> _prefs = SharedPreferences.getInstance();

  // --- ADDED FOR API BASE URL ---
  static const String _baseUrlKey = 'api_base_url'; // Key to store the base URL

  /// Saves the API base URL.
  Future<void> saveBaseUrl(String baseUrl) async {
    final prefs = await _prefs; // Await the Future to get the SharedPreferences instance
    await prefs.setString(_baseUrlKey, baseUrl);
  }

  /// Retrieves the API base URL.
  /// Returns null if no base URL has been saved.
  Future<String?> getBaseUrl() async {
    final prefs = await _prefs; // Await the Future
    return prefs.getString(_baseUrlKey);
  }
  // --- END OF ADDED CODE ---


  /// Sets a boolean [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await SharedPreferencesService.instance.setBool('isLoggedIn', true);
  /// ```
  Future<void> setBool(String key, bool value) async {
    final prefs = await _prefs;
    prefs.setBool(key, value);
  }

  /// Sets a string [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await SharedPreferencesService.instance.setString('username', 'Alice');
  /// ```
  Future<void> setString(String key, String value) async {
    final prefs = await _prefs;
    prefs.setString(key, value);
  }

  /// Sets an integer [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await SharedPreferencesService.instance.setInt('age', 30);
  /// ```
  Future<void> setInt(String key, int value) async {
    final prefs = await _prefs;
    prefs.setInt(key, value);
  }

  /// Sets a list of strings [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await SharedPreferencesService.instance.setStringList('favorites', ['Apples', 'Bananas']);
  /// ```
  Future<void> setStringList(String key, List<String> value) async {
    final prefs = await _prefs;
    prefs.setStringList(key, value);
  }

  /// Retrieves a boolean value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// bool? isLoggedIn = await SharedPreferencesService.instance.getBool('isLoggedIn');
  /// ```
  Future<bool?> getBool(String key) async {
    final prefs = await _prefs;
    return prefs.getBool(key);
  }

  /// Retrieves a string value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// String? username = await SharedPreferencesService.instance.getString('username');
  /// ```
  Future<String?> getString(String key) async {
    final prefs = await _prefs;
    return prefs.getString(key);
  }

  /// Retrieves an integer value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// int? age = await SharedPreferencesService.instance.getInt('age');
  /// ```
  Future<int?> getInt(String key) async {
    final prefs = await _prefs;
    return prefs.getInt(key);
  }

  /// Retrieves a list of strings for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// List<String>? favorites = await SharedPreferencesService.instance.getStringList('favorites');
  /// ```
  Future<List<String>?> getStringList(String key) async {
    final prefs = await _prefs;
    return prefs.getStringList(key);
  }
}