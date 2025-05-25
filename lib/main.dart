// In lib/main.dart
import 'package:auto_gpt_flutter_client/services/leaderboard_service.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
// import 'package:auto_gpt_flutter_client/views/auth/firebase_auth_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:auto_gpt_flutter_client/views/main_layout.dart';
import 'package:provider/provider.dart';
// import 'package:firebase_core/firebase_core.dart';
// import 'package:firebase_auth/firebase_auth.dart';
import 'package:auto_gpt_flutter_client/constants/app_colors.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/skill_tree_viewmodel.dart';

import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:auto_gpt_flutter_client/services/task_service.dart';
import 'package:auto_gpt_flutter_client/services/benchmark_service.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';

// import 'package:flutter/foundation.dart' show kIsWeb; // No longer strictly needed here if registration is removed
// import 'package:webview_flutter_platform_interface/webview_flutter_platform_interface.dart'; // No longer needed here
// import 'package:webview_flutter_web/webview_flutter_web.dart'; // REMOVE/COMMENT THIS - causing error

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // --- REMOVE OR COMMENT OUT THIS WEBVIEW WEB REGISTRATION BLOCK ---
  // if (kIsWeb) {
  //   try {
  //     WebViewPlatform.instance = WebWebViewPlatform();
  //   } catch (e) {
  //     print('Error setting WebWebViewPlatform in main.dart: $e');
  //   }
  // }
  // --- END WEBVIEW WEB REGISTRATION BLOCK ---

  await SharedPreferencesService.instance.saveBaseUrl("http://localhost:8000");
  runApp(
    MultiProvider(
      providers: [
        Provider<RestApiUtility>(
          create: (context) => RestApiUtility("http://127.0.0.1:8000/ap/v1"),
        ),
        Provider<SharedPreferencesService>(
          create: (context) => SharedPreferencesService.instance,
        ),
        ProxyProvider<RestApiUtility, ChatService>(
          update: (context, restApiUtility, previousChatService) =>
              ChatService(restApiUtility),
        ),
        ProxyProvider2<RestApiUtility, SharedPreferencesService, TaskService>(
          update: (context, restApiUtility, prefsService, previousTaskService) =>
              TaskService(restApiUtility, prefsService),
        ),
        ProxyProvider<RestApiUtility, BenchmarkService>(
          update: (context, restApiUtility, previousBenchmarkService) =>
              BenchmarkService(restApiUtility),
        ),
        ProxyProvider<RestApiUtility, LeaderboardService>(
          update: (context, restApiUtility, previousLeaderboardService) =>
              LeaderboardService(restApiUtility),
        ),
        ChangeNotifierProxyProvider2<RestApiUtility, SharedPreferencesService,
            SettingsViewModel>(
          create: (context) => SettingsViewModel(
            Provider.of<RestApiUtility>(context, listen: false),
            Provider.of<SharedPreferencesService>(context, listen: false),
          ),
          update: (context, restApiUtility, prefsService, previousSettingsViewModel) =>
              SettingsViewModel(restApiUtility, prefsService),
        ),
        ChangeNotifierProvider<ChatViewModel>(
          create: (context) => ChatViewModel(
            Provider.of<ChatService>(context, listen: false),
            Provider.of<TaskService>(context, listen: false),
            Provider.of<SharedPreferencesService>(context, listen: false),
          ),
        ),
        ChangeNotifierProvider<TaskViewModel>(
          create: (context) => TaskViewModel(
            Provider.of<TaskService>(context, listen: false),
            Provider.of<SharedPreferencesService>(context, listen: false),
          )..fetchAndCombineData(),
        ),
        ChangeNotifierProvider<SkillTreeViewModel>(
            create: (context) => SkillTreeViewModel()),
        ChangeNotifierProvider<TaskQueueViewModel>(
          create: (context) => TaskQueueViewModel(
            Provider.of<BenchmarkService>(context, listen: false),
            Provider.of<LeaderboardService>(context, listen: false),
            Provider.of<SharedPreferencesService>(context, listen: false),
          ),
        ),
      ],
      child: const MyApp(), // Added const
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key); // Added const

  @override
  Widget build(BuildContext context) {
    // Define the light theme color scheme
    const ColorScheme lightColorScheme = ColorScheme(
      brightness: Brightness.light,
      primary: AppColors.primaryLight,
      onPrimary: AppColors.whiteFillLight,
      secondary: AppColors.accent5Light,
      onSecondary: AppColors.whiteFillLight,
      error: AppColors.accentDeniedLight,
      onError: AppColors.whiteFillLight,
      background: AppColors.skilltreeBackgroundLight,
      onBackground: AppColors.ultraWhiteLight,
      surface: AppColors.cardLight,
      onSurface: AppColors.ultraWhiteLight,
    );

    final ThemeData lightTheme = ThemeData.light().copyWith(
      colorScheme: lightColorScheme,
      primaryColor: lightColorScheme.primary,
      scaffoldBackgroundColor: lightColorScheme.background,
      cardColor: lightColorScheme.surface,
      appBarTheme: AppBarTheme(
        backgroundColor: lightColorScheme.primary,
        foregroundColor: lightColorScheme.onPrimary,
        elevation: 0.5,
      ),
    );

    final ColorScheme darkColorScheme = ColorScheme(
      brightness: Brightness.dark,
      primary: AppColors.adjustedPrimaryDark,
      onPrimary: AppColors.lightOnDarkUIText,
      secondary: AppColors.accent5Dark,
      onSecondary: AppColors.lightOnDarkUIText,
      error: AppColors.accentDeniedDark,
      onError: AppColors.lightOnDarkUIText,
      background: AppColors.darkGreyUI,
      onBackground: AppColors.lightOnDarkUIText,
      surface: AppColors.midGreyUI,
      onSurface: AppColors.lightOnDarkUIText,
    );

    final ThemeData darkGrayTheme = ThemeData.dark().copyWith(
      colorScheme: darkColorScheme,
      primaryColor: darkColorScheme.primary,
      scaffoldBackgroundColor: darkColorScheme.background,
      canvasColor: AppColors.darkerGreyUI,
      cardColor: darkColorScheme.surface,
      dialogBackgroundColor: darkColorScheme.surface,
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.darkerGreyUI,
        elevation: 0,
        iconTheme: IconThemeData(color: darkColorScheme.onSurface),
        titleTextStyle: TextStyle(color: darkColorScheme.onSurface, fontSize: 20, fontWeight: FontWeight.w500),
      ),
      drawerTheme: DrawerThemeData(
        backgroundColor: AppColors.darkerGreyUI,
      ),
      iconTheme: IconThemeData(color: darkColorScheme.onSurface.withOpacity(0.85)),
      textTheme: ThemeData.dark().textTheme.apply(
            bodyColor: darkColorScheme.onSurface,
            displayColor: darkColorScheme.onSurface,
          ).copyWith(
        bodyLarge: TextStyle(color: darkColorScheme.onSurface),
        bodyMedium: TextStyle(color: darkColorScheme.onSurface.withOpacity(0.85)),
        titleLarge: TextStyle(color: darkColorScheme.onSurface, fontWeight: FontWeight.w600),
        titleMedium: TextStyle(color: darkColorScheme.onSurface, fontWeight: FontWeight.w500),
        labelLarge: TextStyle(color: darkColorScheme.onPrimary, fontWeight: FontWeight.w600),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: darkColorScheme.primary,
          foregroundColor: darkColorScheme.onPrimary,
          textStyle: TextStyle(color: darkColorScheme.onPrimary, fontWeight: FontWeight.w600),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: darkColorScheme.primary,
        )
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: darkColorScheme.onSurface,
          side: BorderSide(color: darkColorScheme.onSurface.withOpacity(0.3)),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.darkerGreyUI,
        hintStyle: TextStyle(color: darkColorScheme.onSurface.withOpacity(0.5)),
        labelStyle: TextStyle(color: darkColorScheme.onSurface.withOpacity(0.8)),
        iconColor: darkColorScheme.onSurface.withOpacity(0.7),
        prefixIconColor: darkColorScheme.onSurface.withOpacity(0.7),
        suffixIconColor: darkColorScheme.onSurface.withOpacity(0.7),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8.0),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8.0),
          borderSide: BorderSide(color: darkColorScheme.surface.withOpacity(0.5)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8.0),
          borderSide: BorderSide(color: darkColorScheme.primary, width: 1.5),
        ),
        counterStyle: TextStyle(color: darkColorScheme.onSurface.withOpacity(0.6)),
      ),
      listTileTheme: ListTileThemeData(
        iconColor: darkColorScheme.onSurface.withOpacity(0.8),
        textColor: darkColorScheme.onSurface,
        selectedTileColor: darkColorScheme.primary.withOpacity(0.15),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: AppColors.darkerGreyUI,
        selectedItemColor: darkColorScheme.primary,
        unselectedItemColor: darkColorScheme.onSurface.withOpacity(0.6),
      ),
      dialogTheme: DialogTheme(
        backgroundColor: darkColorScheme.surface,
        titleTextStyle: TextStyle(color: darkColorScheme.onSurface, fontSize: 18, fontWeight: FontWeight.bold),
        contentTextStyle: TextStyle(color: darkColorScheme.onSurface.withOpacity(0.9)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      popupMenuTheme: PopupMenuThemeData(
        color: AppColors.darkerGreyUI,
        textStyle: TextStyle(color: darkColorScheme.onSurface),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: AppColors.darkerGreyUI.withOpacity(0.95),
          borderRadius: BorderRadius.circular(4),
        ),
        textStyle: TextStyle(color: darkColorScheme.onSurface),
      ),
      dividerColor: darkColorScheme.onSurface.withOpacity(0.12),
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'AutoGPT Flutter Client',
      theme: lightTheme,
      darkTheme: darkGrayTheme,
      themeMode: ThemeMode.dark,
      builder: (context, child) {
        final currentTheme = Theme.of(context);
        return CupertinoTheme(
          data: CupertinoThemeData(brightness: currentTheme.brightness),
          child: child!,
        );
      },
      home: MainLayout(),
    );
  }
}